import time
import datetime

import torch
from torch.nn import functional as F
from utils import *
from utils.dataset import *
from utils.evaluator import Classification
import numpy as np


def discrepancy(y1, y2):
    return (y1 - y2).abs().mean()


class CoTeachingTrainer:
    def __init__(self, models, optimizer, device, index, source_domain, **params):
        self.index = index
        self._models = models
        self.F1 = models["F1"]
        self.F2 = models["F2"]
        self.C1 = models["C1"]
        self.C2 = models["C2"]
        self._optims = optimizer
        self.source_domain = source_domain

        self.device = device
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.batch_size = params['batch_size']
        self.forget_rate = params['forget_rate']
        self.folder = osp.join(params['save_model_addr'], f"coTeaching{index}")

        self.max_epoch = 0
        self.start_epoch = self.epoch = 0
        self.evaluator = Classification()
        mkdir_if_missing(self.folder)

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]['lr']

    def set_model_mode(self, mode='train', names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == 'train':
                self._models[name].train()
            else:
                self._models[name].eval()

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Loss is infinite or NaN!')

    def model_backward(self, loss, retain_graph):
        self.detect_anomaly(loss)
        loss.backward(retain_graph=retain_graph)

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None, retain_graph=False):
        self.model_zero_grad(names)
        self.model_backward(loss, retain_graph)
        self.model_update(names)

    def save_model(self, epoch, directory, is_best=False, model_name=''):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            save_checkpoint(
                {
                    'state_dict': model_dict,
                    'epoch': epoch + 1,
                    'optimizer': optim_dict
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name
            )

    def after_epoch(self, test_loader):
        if (self.epoch + 1) % 1 == 0:
            self.test(test_loader)
        last_epoch = (self.epoch + 1) == self.max_epoch
        if last_epoch:
            self.save_model(self.epoch, self.folder)

    def before_train(self):
        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print(f'Finished CoTeaching_{self.index} training')

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed: {}'.format(elapsed))

    def train_dg(self, train_loader, test_loader, max_epoch):
        self.max_epoch = max_epoch
        self.rate_schedule = np.ones(self.max_epoch) * self.forget_rate
        self.rate_schedule[:10] = np.linspace(0, self.forget_rate ** 1, 10)
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.run_epoch(train_loader)
            self.after_epoch(test_loader)
        self.after_train()

    def run_epoch(self, train_loader):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader = len(train_loader)
        num_batches  = len_train_loader

        train_loader_iter = iter(train_loader)

        end = time.time()

        for self.batch_idx in range(len_train_loader):
            try:
                batch_train = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(train_loader)
                batch_train = next(train_loader_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_train)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (self.batch_idx + 1) % 10 == 0:
                nb_this_epoch = num_batches - (self.batch_idx + 1)
                nb_future_epochs = (
                                           self.max_epoch - (self.epoch + 1)
                                   ) * num_batches
                eta_seconds = batch_time.avg * (nb_this_epoch + nb_future_epochs)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'epoch [{0}/{1}][{2}/{3}]\t'
                    'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'eta {eta}\t'
                    '{losses}\t'
                    'lr {lr}'.format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr()
                    )
                )

            end = time.time()

    def ELogLoss(self, probx):  #没有加上负号
        '''
        x : [B, X]
        '''
        logx = torch.log(probx + 1e-5)
        # [B, X] => [B, 1, X]
        probx = probx.unsqueeze(1)
        # [B, X] => [B, X, 1]
        logx = logx.unsqueeze(2)
        # [B, 1, 1]
        res = torch.bmm(probx, logx)
        # [B, 1]
        return res.squeeze(2)

    def forward_backward(self, batch):
        parsed = self.parse_batch_train(batch)
        input, label, paths = parsed

        output1 = self.F1(input)
        output1 = self.C1(output1)
        output2 = self.F2(input)
        output2 = self.C2(output2)

        domain_true = []
        domain_pse = []
        for i,path in enumerate(paths):
            if path.find(self.source_domain) != -1:
                domain_true.append(i)
            else:
                domain_pse.append(i)


        domain_true = torch.tensor(domain_true).type(torch.long)
        domain_pse = torch.tensor(domain_pse).type(torch.long)
        output1_true = output1[domain_true]
        output2_true = output2[domain_true]
        label_true = label[domain_true]

        output1_pse = output1[domain_pse]
        output2_pse = output2[domain_pse]
        label_pse = label[domain_pse]

        loss_1_true = F.cross_entropy(output1_true, label_true)
        loss_2_true = F.cross_entropy(output2_true, label_true)

        loss_1 = F.cross_entropy(output1_pse, label_pse, reduction='none')
        ind_1_sorted = np.argsort(loss_1.data.cpu())
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(output2_pse, label_pse, reduction='none')
        ind_2_sorted = np.argsort(loss_2.data.cpu())
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - self.rate_schedule[self.epoch]
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]

        loss_1_update = F.cross_entropy(output1_pse[ind_2_update], label_pse[ind_2_update])  # 交换
        loss_2_update = F.cross_entropy(output2_pse[ind_1_update], label_pse[ind_1_update])

        # information maximization loss
        pred1_pse = F.softmax(output1_pse, 1)
        pred2_pse = F.softmax(output2_pse, 1)
        pred1_pse_hat = pred1_pse.mean(dim=0).unsqueeze(0)
        loss_1_im = -self.ELogLoss(pred1_pse).mean(0) + self.ELogLoss(pred1_pse_hat).sum()
        pred2_pse_hat = pred2_pse.mean(dim=0).unsqueeze(0)
        loss_2_im = -self.ELogLoss(pred2_pse).mean(0) + self.ELogLoss(pred2_pse_hat).sum()

        if domain_true.shape != torch.Size([0]):
            loss_1_final = loss_1_true + loss_1_update + loss_1_im
            loss_2_final = loss_2_true + loss_2_update + loss_2_im
        else:
            loss_1_final = loss_1_update + loss_1_im
            loss_2_final = loss_2_update + loss_2_im
        self.model_backward_and_update(loss_1_final, ['F1','C1'])
        self.model_backward_and_update(loss_2_final, ['F2','C2'])

        loss_summary = {
            'loss_model1_true': loss_1_true.item(),
            'loss_model2_true': loss_2_true.item(),
            'loss_model1_pse': loss_1_update.item(),
            'loss_mode2_pse': loss_2_update.item(),
            'loss_1_final': loss_1_final.item(),
            'loss_2_final': loss_2_final.item()
        }

        return loss_summary


        return

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        impath = batch['impath']

        input = input.to(self.device)
        label = label.to(self.device)


        return input, label, impath


    def parse_batch_test(self, batch):
        input = batch['img']
        label = batch['label']
        impath = batch["impath"]

        input = input.to(self.device)
        label = label.to(self.device)


        return input, label, impath

    def model_inference(self, input):
        feat = self.F1(input)
        return self.C1(feat)

    def load_model(self, directory, epoch=None):
        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models[name].load_state_dict(state_dict)

    def test(self, test_loader):
        """A testing pipeline."""
        self.set_model_mode('eval')
        self.evaluator.reset()

        for batch_idx, batch in enumerate(test_loader):
            input, label, _ = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        return results['accuracy']

    def update_lr(self, index=1):
        for key in self._optims:
            opt = self._optims[key]
            lr = opt.defaults["lr"] / index**2
            opt.param_groups[0]['lr'] = lr

    def get_pl(self, target_loader, ratio):
        pl = []
        self.set_model_mode('eval')
        for batch_idx, batch in enumerate(target_loader):
            input, label, impath = self.parse_batch_test(batch)
            output1 = self.F1(input)
            output1 = self.C1(output1)
            output2 = self.F2(input)
            output2 = self.C2(output2)
            pred1 = F.softmax(output1, 1)  # 128*10
            pred2 = F.softmax(output2, 1)  # 128*10
            _, p_label_1 = torch.max(pred1, 1)  # 1 3 3 5
            _, p_label_2 = torch.max(pred2, 1)  # 2 3 4 5
            agree_id = torch.where(p_label_1 == p_label_2)[0].cpu().numpy()  # 找到相同的下标
            label = label[agree_id]  # 切片
            output1 = output1[agree_id]
            output2 = output2[agree_id]
            loss = F.cross_entropy(output1, label, reduction='none')
            ind_sorted = np.argsort(loss.data.cpu())
            loss_sorted = loss[ind_sorted]
            num_remember = int(ratio * len(loss_sorted))
            ind_update_clean = ind_sorted[:num_remember]
            ind_update_clean = agree_id[ind_update_clean]
            ind_update_hard = ind_sorted[num_remember:]
            ind_update_hard = agree_id[ind_update_hard]
            pred = pred1.max(1)[1]
            for index, path in enumerate(impath):
                if index in ind_update_clean:
                    pl.append((path, int(pred[index])))
        return pl

