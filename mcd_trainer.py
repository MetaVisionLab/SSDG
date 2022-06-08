import time
import datetime

from torch.nn import functional as F
from utils import *
from utils.dataset import *
from utils.evaluator import Classification

from utils import mixup


def discrepancy(y1, y2):
    return (y1 - y2).abs().mean()


class MCDTrainer:
    def __init__(self, models, optimizer, device, index, **params):
        self.index = index
        self.n_step_f = params["n_step_f"]
        self._models = models
        self.F = models["F"]
        self.C1 = models["C1"]
        self.C2 = models["C2"]
        self._optims = optimizer

        self.device = device
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.batch_size = params['batch_size']
        self.folder = osp.join(params['save_model_addr'], f"MCD{index}")

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
        if (self.epoch + 1) % 10 == 0:
            self.test(test_loader)
        last_epoch = (self.epoch + 1) == self.max_epoch
        if last_epoch:
            self.save_model(self.epoch, self.folder)

    def before_train(self):
        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print(f'Finished MCD_{self.index} training')

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed: {}'.format(elapsed))

    def train_mcd(self, train_loader, test_loader, max_epoch):
        self.max_epoch = max_epoch
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.run_epoch(train_loader, test_loader)
            self.after_epoch(test_loader)
        self.after_train()

    def run_epoch(self, train_loader, test_loader):
        self.set_model_mode('train')
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(train_loader)
        len_train_loader_u = len(test_loader)
        num_batches = min(len_train_loader_x, len_train_loader_u)

        train_loader_x_iter = iter(train_loader)
        train_loader_u_iter = iter(test_loader)

        end = time.time()

        for self.batch_idx in range(num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(train_loader)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(test_loader)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
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

    def forward_backward(self, batch_x, batch_u):
        parsed = self.parse_batch_train(batch_x, batch_u)
        input_x, label_x_a, label_x_b, lam, input_u = parsed #mixup

        # Step A
        feat_x = self.F(input_x)
        logit_x1 = self.C1(feat_x)
        logit_x2 = self.C2(feat_x)
        # mixup
        loss_x1 = lam * F.cross_entropy(logit_x1, label_x_a) + (1 - lam) * F.cross_entropy(logit_x1, label_x_b)
        loss_x2 = lam * F.cross_entropy(logit_x2, label_x_a) + (1 - lam) * F.cross_entropy(logit_x2, label_x_b)
        loss_step_A = loss_x1 + loss_x2
        self.model_backward_and_update(loss_step_A)

        # Step B
        with torch.no_grad():
            feat_x = self.F(input_x)
        logit_x1 = self.C1(feat_x)
        logit_x2 = self.C2(feat_x)
        # mixup
        loss_x1 = lam * F.cross_entropy(logit_x1, label_x_a) + (1 - lam) * F.cross_entropy(logit_x1, label_x_b)
        loss_x2 = lam * F.cross_entropy(logit_x2, label_x_a) + (1 - lam) * F.cross_entropy(logit_x2, label_x_b)
        loss_x = loss_x1 + loss_x2

        with torch.no_grad():
            feat_u = self.F(input_u)
        pred_u1 = F.softmax(self.C1(feat_u), 1)
        pred_u2 = F.softmax(self.C2(feat_u), 1)
        loss_dis = discrepancy(pred_u1, pred_u2)

        loss_step_B = loss_x - loss_dis
        self.model_backward_and_update(loss_step_B, ['C1', 'C2'])

        # Step C
        for _ in range(self.n_step_f):
            feat_u = self.F(input_u)
            pred_u1 = F.softmax(self.C1(feat_u), 1)
            pred_u2 = F.softmax(self.C2(feat_u), 1)
            loss_step_C = discrepancy(pred_u1, pred_u2)
            self.model_backward_and_update(loss_step_C, 'F')

        loss_summary = {
            'loss_step_A': loss_step_A.item(),
            'loss_step_B': loss_step_B.item(),
            'loss_step_C': loss_step_C.item()
        }

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x['img']
        label_x = batch_x['label']
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        input_x, label_x_a, label_x_b, lam = mixup_data(input_x, label_x, alpha=1.0)

        return input_x, label_x_a, label_x_b, lam, input_u

    def parse_batch_test(self, batch):
        input = batch['img']
        label = batch['label']
        impath = batch["impath"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label, impath

    def model_inference(self, input):
        feat = self.F(input)
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
            lr = opt.defaults["lr"] / index ** 2
            opt.param_groups[0]['lr'] = lr

    def get_pl(self, target_loader):
        pl = []
        self.set_model_mode('eval')
        for batch_idx, batch in enumerate(target_loader):
            input, _, impath = self.parse_batch_test(batch)
            output = self.model_inference(input)
            pred = output.max(1)[1]
            for index, path in enumerate(impath):
                pl.append((path, int(pred[index])))
        return pl
