import os.path as osp
import PIL
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

DATA_DIR = '/data/liuxin/UPROTO/FEAT/data'
IMAGE_PATH = osp.join(DATA_DIR, 'cub/image')
SPLIT_PATH = osp.join(DATA_DIR, 'cub/split')
CACHE_PATH = osp.join(DATA_DIR, '.cache/')

# This is for the CUB dataset
# It is notable, we assume the cub images are cropped based on the given bounding boxes
# The concept labels are based on the attribute value, which are for further use (and not used in this work)

def identity(x):
    return x


class CUB(Dataset):

    def __init__(self, setname, args, augment=False):
        im_size = args.orig_imsize
        txt_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
        cache_path = osp.join(CACHE_PATH, "{}.{}.{}.pt".format(
            self.__class__.__name__, setname, im_size))

        self.use_im_cache = (im_size != -1)  # not using cache
        if self.use_im_cache:
            if not osp.exists(cache_path):
                print('* Cache miss... Preprocessing {}...'.format(setname))
                resize_ = identity if im_size < 0 else transforms.Resize(
                    im_size)
                data, label = self.parse_csv(txt_path)
                self.data = [resize_(Image.open(path).convert('RGB'))
                             for path in data]
                self.label = label
                print('* Dump cache from {}'.format(cache_path))
                torch.save(
                    {'data': self.data, 'label': self.label}, cache_path)
            else:
                print('* Load cache from {}'.format(cache_path))
                cache = torch.load(cache_path)
                self.data = cache['data']
                self.label = cache['label']
        else:
            self.data, self.label = self.parse_csv(txt_path)

        self.num_class = np.unique(np.array(self.label)).shape[0]
        self.setname = setname
        image_size = 84

        if augment and setname == 'train':
            transforms_list = [
                transforms.RandomResizedCrop(image_size),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            if args.n_aug == 1 or setname == 'val':
                transforms_list = [
                    transforms.Resize(92),
                    transforms.CenterCrop(image_size),
                    # transforms.RandomResizedCrop(image_size),
                    transforms.ToTensor(),
                ]
            else:
                transforms_list = [
                    # transforms.Resize(92),
                    # transforms.CenterCrop(image_size),
                    transforms.RandomResizedCrop(image_size),
                    transforms.ToTensor(),
                ]
        self.n_aug = args.n_aug

        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                         np.array([0.229, 0.224, 0.225]))
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

    def parse_csv(self, txt_path):
        data = []
        label = []
        lb = -1
        self.wnids = []
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            path = osp.join(IMAGE_PATH, wnid, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            label.append(lb)

        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        image=[]
        if self.setname=='train' or self.setname=='val':
            n_aug=1
        else:
            n_aug=self.n_aug
        for augs in range(n_aug):
            if self.use_im_cache:
                image.append(self.transform(data))
            else:
                image.append(self.transform(Image.open(data).convert('RGB')))

        return image, label


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
    trainset = CUB('train', args, augment=args.augment)

