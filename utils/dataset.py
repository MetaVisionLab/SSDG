import os
import os.path as osp
import torch
import torch.utils.data as data
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_image_dirs(root, dname, split):
    root = osp.abspath(root)
    image_dir = osp.join(root, 'images')
    split_dir = osp.join(root, 'splits')

    def read_split_pacs(split_file):
        items = []

        with open(split_file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                impath, label = line.split(' ')
                impath = osp.join(image_dir, impath)
                if 'pacs' in image_dir:
                    label = int(label) - 1
                else:
                    label = int(label)
                items.append((impath, label))

        return items
    if split == 'all':
        file_train = osp.join(
            split_dir, dname + '_train_kfold.txt'
        )
        impath_label_list = read_split_pacs(file_train)
        file_val = osp.join(
            split_dir, dname + '_crossval_kfold.txt'
        )
        impath_label_list += read_split_pacs(file_val)
    else:
        file = osp.join(
            split_dir, dname + '_' + split + '_kfold.txt'
        )
        impath_label_list = read_split_pacs(file)

    return impath_label_list


class base_dataset(data.Dataset):
    def __init__(self, impath_label_list, transform):
        self.impath_label_list = impath_label_list
        self.transform = transform

    def __len__(self):
        return len(self.impath_label_list)

    def __getitem__(self, index):
        impath, label = self.impath_label_list[index]
        img = Image.open(impath).convert('RGB')
        img = self.transform(img)
        output = {
            'img': img,
            'label': torch.tensor(label),
            'impath': impath
        }
        return output

def get_web_image_dirs(root):
    path = os.path.join(root, 'all_list.txt')
    file = open(path, 'r',encoding="gbk")
    items = []
    for line in file.readlines():
        impath = line.split()[0]
        impath = osp.join(root, impath)
        label = int(line.split()[1])
        items.append((impath, label))
    return items
