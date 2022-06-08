import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
from .mixstyle import MixStyle
from torch.nn import functional as F


class Convolution(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x))

class CNN_Digits(nn.Module):

    def __init__(self, c_hidden=64, ms_layers=[], ms_p=0.5, ms_a=0.1):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)
        self.mixstyle = None
        if ms_layers:
            self.mixstyle = MixStyle(p=ms_p, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ['layer1', 'layer2', 'layer3']
            print(f'Insert MixStyle after {ms_layers}')
        self.ms_layers = ms_layers

        self.fdim = 2**2 * c_hidden


    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def forward(self, x):
        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        if 'layer1' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        if 'layer2' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        if 'layer3' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        return x.view(x.size(0), -1)






