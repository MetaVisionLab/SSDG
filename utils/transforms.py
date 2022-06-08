import random
from PIL import Image
from torchvision.transforms import transforms
from utils.gaussian_blur import GaussianBlur


class Random2DTranslation:
    """Randomly translates the input image with a probability.
    Specifically, given a predefined shape (height, width), the
    input is first resized with a factor of 1.125, leading to
    (height*1.125, width*1.125), then a random crop is performed.
    Such operation is done with a probability.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width = int(round(self.width * 1.125))
        new_height = int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)

        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop(
            (x1, y1, x1 + self.width, y1 + self.height)
        )

        return croped_img


class MultiViewDataInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]

    def __call__(self, sample):
        output = [transform(sample) for transform in self.transforms]
        return output


def get_simclr_data_transforms(input_shape, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=eval(input_shape)[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * eval(input_shape)[0])),
                                          transforms.ToTensor()])
    return data_transforms


def strong_transform_train(input_size, type='visda', s=1):
    rotation = transforms.RandomRotation(30)
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    if type == 'digits':
        data_transforms = transforms.Compose([
            rotation,
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.5, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    elif type == 'signs':
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.5, 2.0)),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:  # visda
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
        ])
    return data_transforms


def simple_transform_train(input_size, type=''):
    if type == 'digits':
        data_transforms = transforms.Compose([
            transforms.Resize([input_size, input_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            Random2DTranslation(input_size, input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
        ])
    return data_transforms




def simple_transform_test(input_size, type='visda'):
    if type == 'digits' or type == 'signs':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
    data_transforms = transforms.Compose([transforms.Resize([input_size, input_size]),
                                          transforms.ToTensor(),
                                          normalize])
    return data_transforms


def multiview_transform(input_size, padding, type='visda'):
    totensor = transforms.ToTensor()
    if type == 'digits' or type == 'signs':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
    if padding >= 0:  # up-scale
        data_transforms = transforms.Compose([
            transforms.Resize([input_size + 2 * padding, input_size + 2 * padding]),
            transforms.CenterCrop([input_size, input_size]),
            totensor,
            normalize
        ])
    else:  # down-scale
        data_transforms = transforms.Compose([
            transforms.Resize([input_size, input_size]),
            transforms.Pad(-padding, fill=0, padding_mode='constant'),
            transforms.Resize([input_size, input_size]),
            totensor,
            normalize
        ])
    return data_transforms
