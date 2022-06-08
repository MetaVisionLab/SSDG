from .resnet import *
from .cnn_digitsdg import *

models = {
	'R18': resnet18,
	'CNN':CNN_Digits
}