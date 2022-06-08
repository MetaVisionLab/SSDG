import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
from .mixstyle import MixStyle

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class resnet18(nn.Module):
    def __init__(self, pretrained=True,ms_layers=[],ms_p=0.5,ms_a=0.1):
        super(resnet18, self).__init__()
        self.model = torchvision.models.resnet18(pretrained=pretrained)
        self.fdim = self.model.fc.in_features
        self.mixstyle = None
        if ms_layers:
            self.mixstyle = MixStyle(p = ms_p, alpha= ms_a)
            for layer_name in ms_layers:
                assert layer_name in ['layer1', 'layer2', 'layer3']
            print(f'Insert MixStyle after {ms_layers}')
        self.ms_layers = ms_layers

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        if 'layer1' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.model.layer2(x)
        if 'layer2' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.model.layer3(x)
        if 'layer3' in self.ms_layers:
            x = self.mixstyle(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def init_pretrained_weights(self, model_url):
        pretrain_dict = model_zoo.load_url(model_url)
        pretrain_dict_1 = {}
        for key in pretrain_dict:
            if "model." in key:
                pretrain_dict_1[key.replace("model.", "")] = pretrain_dict[key]
        self.model.load_state_dict(pretrain_dict_1, strict=False)

    def load_imagenet_dict(self, pretrained_dict):
        print('--------Loading weight--------')
        model_dict = self.model.state_dict()
        if list(model_dict.keys())[0].find('model') != -1:
            pretrained_dict = {'model.' + k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)


