import os
import torch
import yaml
import numpy as np
import os.path as osp
from torchvision import datasets
from utils.transforms import *
from utils.dataset import *
from utils import *
from torch.utils.data.dataloader import DataLoader
from models import models
from mcd_trainer import MCDTrainer
from coteaching_trainer import CoTeachingTrainer

from models.build import build_mcd_model,build_dg_model,build_digits_dg_model,build_digits_mcd_model

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_mcd_optimizer(model, config):
    if config['optimizer']['name'] == 'SGD':
        opt_F = torch.optim.SGD(list(model["F"].parameters()),
                                **config['optimizer']['params'])
        opt_C1 = torch.optim.SGD(list(model["C1"].parameters()),
                                 **config['optimizer']['params'])
        opt_C2 = torch.optim.SGD(list(model["C2"].parameters()),
                                 **config['optimizer']['params'])
    else:
        opt_F = torch.optim.Adam(list(model["F"].parameters()),
                                 **config['optimizer']['params'])
        opt_C1 = torch.optim.Adam(list(model["C1"].parameters()),
                                  **config['optimizer']['params'])
        opt_C2 = torch.optim.Adam(list(model["C2"].parameters()),
                                  **config['optimizer']['params'])

    return {"F": opt_F, "C1": opt_C1, "C2": opt_C2}

def build_dg_optimizer(model, config):
    if config['optimizer']['name'] == 'SGD':
        opt_F1 = torch.optim.SGD(list(model["F1"].parameters()),
                                **config['optimizer']['params'])
        opt_F2 = torch.optim.SGD(list(model["F2"].parameters()),
                                 **config['optimizer']['params'])
        opt_C1 = torch.optim.SGD(list(model["C1"].parameters()),
                                 **config['optimizer']['params'])
        opt_C2 = torch.optim.SGD(list(model["C2"].parameters()),
                                 **config['optimizer']['params'])
    else:
        opt_F1 = torch.optim.Adam(list(model["F1"].parameters()),
                                 **config['optimizer']['params'])
        opt_F2 = torch.optim.Adam(list(model["F2"].parameters()),
                                  **config['optimizer']['params'])
        opt_C1 = torch.optim.Adam(list(model["C1"].parameters()),
                                  **config['optimizer']['params'])
        opt_C2 = torch.optim.Adam(list(model["C2"].parameters()),
                                  **config['optimizer']['params'])

    return {"F1": opt_F1,"F2": opt_F2, "C1": opt_C1, "C2": opt_C2 }


def build_dataloader(data_list, batch_size, config, transform=None, istrain=True):
    dataset = base_dataset(data_list, transform=transform)
    return DataLoader(dataset, batch_size=batch_size,
                      num_workers=config['trainer']['num_workers'],
                      drop_last=istrain, shuffle=istrain)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='visda', help='task name')
    args = parser.parse_args()

    set_random_seed(1)

    config = yaml.load(open("./config/" + args.task + ".yaml", "r"), Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # logger info
    logdir = osp.join(osp.dirname(__file__), config['log']['save_addr'], config['log']['save_name'] + '.log')
    setup_logger(logdir)
    print(config)

    # network
    mcd_model_1 = build_mcd_model(config).to(device)
    mcd_model_2 = build_mcd_model(config).to(device)
    dg_model = build_dg_model(config).to(device)

    # optim
    mcd_opt_1 = build_mcd_optimizer(mcd_model_1, config)
    mcd_opt_2 = build_mcd_optimizer(mcd_model_2, config)
    dg_opt = build_dg_optimizer(dg_model, config)


    # data_list, transform
    input_size = config['data_transforms']['input_size']
    batch_size = config['trainer']['batch_size']
    mcd_transform_test = simple_transform_test(input_size=input_size, type = config['data']['type'])
    mcd_transform_train = simple_transform_train(input_size=input_size, type = config['data']['type'])
    impath_label_x = get_image_dirs(root=config['data']['root'],
                                    dname=config['data']['source_domain_x'],
                                    split="train")  #源域1
    impath_label_u_1 = get_image_dirs(root=config['data']['root'],
                                      dname=config['data']['source_domain_u_1'],
                                      split="train")  #源域2
    impath_label_u_2 = get_image_dirs(root=config['data']['root'],
                                      dname=config['data']['source_domain_u_2'],
                                      split="train")  #源域3
    impath_label_t = get_image_dirs(root=config['data']['root'],
                                    dname=config['data']['target_domain'],
                                    split="all")
    fake_mcd_u_1 = []
    fake_mcd_u_2 = []
    fake_dg_u_1 = []
    fake_dg_u_2 = []

    # trainer
    trainer_mcd_1 = MCDTrainer(mcd_model_1, mcd_opt_1, device, 1, **config['trainer'])
    trainer_mcd_2 = MCDTrainer(mcd_model_2, mcd_opt_2, device, 2, **config['trainer'])

    for index in range(3):
        print(f"Round {index}: Training MCD.".center(100, "#"))
        # dataloader
        train_data_1 = impath_label_x + fake_dg_u_1
        train_data_2 = impath_label_x + fake_dg_u_2
        dataloader_x_1 = build_dataloader(train_data_1, batch_size, config, mcd_transform_train)
        print('dataset dataloader_x_1: {}'.format(len(dataloader_x_1)))
        dataloader_x_2 = build_dataloader(train_data_2, batch_size, config, mcd_transform_train)
        print('dataset dataloader_x_2: {}'.format(len(dataloader_x_2)))
        dataloader_u_1 = build_dataloader(impath_label_u_1, batch_size, config, mcd_transform_train)
        print('dataset dataloader_u_1: {}'.format(len(dataloader_u_1)))
        dataloader_u_2 = build_dataloader(impath_label_u_2, batch_size, config, mcd_transform_train)
        print('dataset dataloader_u_2: {}'.format(len(dataloader_u_2)))

        # train
        trainer_mcd_1.update_lr(index + 1)
        trainer_mcd_2.update_lr(index + 1)
        trainer_mcd_1.train_mcd(dataloader_x_1, dataloader_u_1, 30)
        trainer_mcd_2.train_mcd(dataloader_x_2, dataloader_u_2, 30)
        del dataloader_x_1, dataloader_x_2, dataloader_u_1, dataloader_u_2

        # test dataloader.
        dataloader_u_1 = build_dataloader(impath_label_u_1, batch_size, config, mcd_transform_test, False)
        dataloader_u_2 = build_dataloader(impath_label_u_2, batch_size, config, mcd_transform_test, False)
        # test
        print("test dataloader_u_1.".center(60, "#"))
        trainer_mcd_1.test(dataloader_u_1)
        print("test dataloader_u_2.".center(60, "#"))
        trainer_mcd_2.test(dataloader_u_2)
        # get pseudo label.
        print("get pseudo label.".center(60, "#"))
        fake_mcd_u_1 = trainer_mcd_1.get_pl(dataloader_u_1)
        fake_mcd_u_2 = trainer_mcd_2.get_pl(dataloader_u_2)
        del dataloader_u_1, dataloader_u_2

        # Train Co-teaching
        print("Training CO-Teaching.".center(100, "#"))
        train_data = impath_label_x + fake_mcd_u_1 + fake_mcd_u_2
        dg_transform_test = simple_transform_test(input_size=input_size, type = config['data']['type'])
        dg_transform_train = simple_transform_train(input_size=input_size, type = config['data']['type'])
        dataloader_train = build_dataloader(train_data, batch_size, config, dg_transform_train)
        print('dataset dg train dataloader: {}'.format(len(dataloader_train)))
        dataloader_test = build_dataloader(impath_label_t, batch_size, config, dg_transform_test)
        print('dataset dg test dataloader: {}'.format(len(dataloader_test)))

        source_domain = config['data']['source_domain_x']
        trainer_dg = CoTeachingTrainer(dg_model, dg_opt, device, 1, source_domain, **config['trainer'])
        #train
        trainer_dg.update_lr(index + 1)
        trainer_dg.train_dg(dataloader_train,dataloader_test,15)

        #test dataloader
        dataloader_u_1 = build_dataloader(impath_label_u_1, batch_size, config, mcd_transform_test, False)
        dataloader_u_2 = build_dataloader(impath_label_u_2, batch_size, config, mcd_transform_test, False)

        #get pseudo label
        ratio = config['ratio']
        print("get dg pseudo label.".center(60, "#"))
        fake_dg_u_1 = trainer_dg.get_pl(dataloader_u_1,ratio = ratio)
        fake_dg_u_2 = trainer_dg.get_pl(dataloader_u_2,ratio = ratio)
        del dataloader_u_1, dataloader_u_2



if __name__ == '__main__':
    main()
