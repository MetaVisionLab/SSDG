from models import models
import torch.nn as nn


def build_mcd_model(config):
    num_classes = config['data']['class_number']
    F = models["R18"]()
    fdim = F.fdim
    C1 = nn.Linear(fdim, num_classes)
    C2 = nn.Linear(fdim, num_classes)
    model = nn.ModuleDict({"F": F, "C1": C1, "C2": C2})
    return model


def build_dg_model(config):
    num_classes = config['data']['class_number']
    F1 = models["R18"](ms_layers=['layer1', 'layer2', 'layer3'])
    F2 = models["R18"](ms_layers=['layer1', 'layer2', 'layer3'])
    fdim = F1.fdim
    C1 = nn.Linear(fdim, num_classes)
    C2 = nn.Linear(fdim, num_classes)
    model = nn.ModuleDict({"F1": F1, "F2": F2, "C1": C1, "C2": C2})
    return model

def build_digits_mcd_model(config):   #mcd用cnn
    num_classes = config['data']['class_number']
    F = models["CNN"]()
    fdim = F.fdim
    # init_network_weights(F, init_type='kaiming')
    C1 = nn.Linear(fdim, num_classes)
    C2 = nn.Linear(fdim, num_classes)
    model = nn.ModuleDict({"F": F, "C1": C1, "C2": C2})
    return model

def build_digits_dg_model(config):  #digits也用cnn
    num_classes = config['data']['class_number']
    F1 = models["CNN"](ms_layers=['layer1', 'layer2', 'layer3'])
    F2 = models["CNN"](ms_layers=['layer1', 'layer2', 'layer3'])
    fdim = F1.fdim
    # init_network_weights(F1, init_type='kaiming')
    # init_network_weights(F2, init_type='kaiming')
    C1 = nn.Linear(fdim, num_classes)
    C2 = nn.Linear(fdim, num_classes)
    model = nn.ModuleDict({"F1": F1, "F2": F2, "C1": C1, "C2": C2})
    return model



def init_network_weights(model, init_type='normal', gain=0.02):

    def _init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (
            classname.find('Conv') != -1 or classname.find('Linear') != -1
        ):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method {} is not implemented'.
                    format(init_type)
                )
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm') != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('InstanceNorm') != -1:
            if m.weight is not None and m.bias is not None:
                nn.init.constant_(m.weight.data, 1.0)
                nn.init.constant_(m.bias.data, 0.0)

    model.apply(_init_func)