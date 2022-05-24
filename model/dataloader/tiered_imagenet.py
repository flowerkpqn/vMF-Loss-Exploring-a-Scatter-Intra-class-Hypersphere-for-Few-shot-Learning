from __future__ import print_function

import os
import os.path as osp
import numpy as np
import pickle
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH1, 'data/tieredimagenet/')
SPLIT_PATH = osp.join(ROOT_PATH2, 'data/miniimagenet/split')

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(
                img_path))
            pass
    return img


file_path = {'train': [os.path.join(IMAGE_PATH, 'train_images.npz'), os.path.join(IMAGE_PATH, 'train_labels.pkl')],
             'val': [os.path.join(IMAGE_PATH, 'val_images.npz'), os.path.join(IMAGE_PATH, 'val_labels.pkl')],
             'test': [os.path.join(IMAGE_PATH, 'test_images.npz'), os.path.join(IMAGE_PATH, 'test_labels.pkl')]}


class tieredImageNet(data.Dataset):
    def __init__(self, setname, args, augment=False):
        assert(setname == 'train' or setname == 'val' or setname == 'test')
        # image_path = file_path[setname][0]
        # label_path = file_path[setname][1]

        # data_train = load_data(label_path)
        # labels = data_train['labels']
        # self.data = np.load(image_path)['images']
        # label = []
        # lb = -1
        # self.wnids = []
        # for wnid in labels:
        #     if wnid not in self.wnids:
        #         self.wnids.append(wnid)
        #         lb += 1
        #     label.append(lb)
        self.train_dir =  os.path.join(DATA_DIR,setname)
        self.data, self.label = self._process_dir(self.train_dir)
        self.num_class = len(set(self.label))
        self.setname = setname

        if augment and setname == 'train':
            transforms_list = [
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            if args.n_aug == 1 or setname == 'val':
                transforms_list = [
                    transforms.Resize(92),
                    transforms.CenterCrop(84),
                    # transforms.RandomResizedCrop(84),
                    transforms.ToTensor(),
                ]

            transforms_list = [
                transforms.Resize(84),
                transforms.ToTensor(),
            ]
            self.n_aug = args.n_aug
        # Transformation
        if args.backbone_class == 'ConvNet'  or args.backbone_class == 'ConvNet6':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                         np.array([0.229, 0.224, 0.225]))
                ])
        elif args.backbone_class == 'ResNet':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                         np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
                ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                         np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
                ])
        elif args.backbone_class == 'Res18':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
        elif args.backbone_class == 'WRN':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
        else:
            raise ValueError(
                'Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]
        image=[]
        if self.setname=='train' or self.setname=='val':
            n_aug=1
        else:
            n_aug=self.n_aug
        for augs in range(n_aug):
            image.append(self.transform(read_image(data)))
        return image, label

    def __len__(self):
        return len(self.data)

    def _process_dir(self, dir_path):
        cat_container = sorted(os.listdir(dir_path))
        catname2label = {cat: label for label, cat in enumerate(cat_container)}

        dataset = []
        labels = []
        for cat in cat_container:
            for img_path in sorted(os.listdir(os.path.join(dir_path, cat))):
                if '.jpg' not in img_path:
                    continue
                label = catname2label[cat]
                dataset.append(os.path.join(dir_path, cat, img_path))
                labels.append(label)
        return dataset, labels


if __name__ == "__main__":
    import sys
    print(sys.path)
    sys.path.append('/data/liuxin/UPROTO/FEAT')
    from model.utils import (
        pprint, set_gpu,
        get_command_line_parser,
        postprocess_args
    )
    parser = get_command_line_parser()
    args = postprocess_args(parser.parse_args())
    trainset = tieredImageNet('train', args, augment=args.augment)
    trainiter = iter(trainset).__next__()
    print("ds")
