import os
import torch
import yaml
import numpy as np
import os.path as osp
import pdb
from torchvision import datasets
from utils.transforms import *
from utils.dataset import *
from torch.utils.data.dataloader import DataLoader
from models import models


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='visda', help='task name')
    args = parser.parse_args()

    config = yaml.load(open("./config/" + args.task + ".yaml", "r"), Loader=yaml.FullLoader)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(config)
    print(f"Training with: {device}")

    # network
    network = models[config['network']['name']](class_number=config['data']['class_number']).to(device)
    # pretrained_folder = config['network']['fine_tune_from']
    model_name = ['resnet101_source_epoch_1', 'resnet101_source_epoch_2']
    for i in range(len(model_name)):
        pretrained_folder = osp.join('checkpoints/visda', model_name[i] + '.pth')
        if pretrained_folder:
            try:
                load_params = torch.load(pretrained_folder)
                network.load_state_dict(load_params['network_state_dict'], strict=True)
            except FileNotFoundError:
                print("Pre-trained weights not found. Training from scratch.")

        # optim
        if config['optimizer']['name'] == 'SGD':
            optimizer = torch.optim.SGD(list(network.parameters()),
                                        **config['optimizer']['params'])
        else:
            optimizer = torch.optim.Adam(list(network.parameters()),
                                         **config['optimizer']['params'])

        # dataloader
        input_size = config['data_transforms']['input_size']
        batch_size = config['trainer']['batch_size']
        data_transform = simple_transform_test(input_size=input_size, type=config['data'][
            'type'])  # must be test_transform for better clustering
        imagedirs, gt_labels = get_image_dirs(root=config['data']['trainset'])
        train_dataset = base_dataset(imagedirs, labels=gt_labels, transform=data_transform)
        print('trainset {}: {}'.format(config['data']['trainset'], len(train_dataset)))

        test_imagedirs, test_labels = get_image_dirs(root=config['data']['testset'])

        test_dataset = base_dataset(test_imagedirs, test_labels, transform=data_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=config['trainer']['num_workers'], drop_last=False, shuffle=False)
        print('testset {}: {}'.format(config['data']['testset'], len(test_dataset)))

        # trainer
        trainer = ADATrainer(network=network,
                             optimizer=optimizer,
                             device=device,
                             params=config)

        # AdaBN
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=config['trainer']['num_workers'], drop_last=False,
                                  shuffle=True)  # must be shuffle
        acc_list, acc = trainer.test_class_wise(test_loader)

        print('model name: {}'.format(model_name[i]))
        for x in acc_list:
            print('before adapt-bn accuracy: {:.4f}'.format(x))
        print('before adapt-bn average accuracy: {:.4f}'.format(acc))

        trainer.update_bn(train_loader)
        acc_list, acc = trainer.test_class_wise(test_loader)
        print('model name: {}'.format(model_name[i]))
        for x in acc_list:
            print('after adapt-bn accuracy: {:.4f}'.format(x))
        print('after adapt-bn average accuracy: {:.4f}'.format(acc))
        trainer.save_model(osp.join('checkpoints/visda', model_name[i] + '_af_adabn.pth'))


if __name__ == '__main__':
    main()
