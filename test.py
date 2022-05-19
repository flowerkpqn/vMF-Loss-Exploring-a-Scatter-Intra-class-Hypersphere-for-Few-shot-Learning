#-*- coding=utf-8 -*-
#@Time:7/5/2022 下午 2:15
#@Author:kpqn
#@File:test.py
#@software:PyCharm

import time
import os.path as osp
import numpy as np

import torch

from model.utils import (
    count_acc, compute_confidence_interval,
)
from tqdm import tqdm

def evaluate_test(self):
    # restore model args
    args = self.args
    # evaluation mode
    self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
    self.model.eval()
    record = np.zeros((10000, 2))  # loss and acc
    label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
    label = label.type(torch.LongTensor)
    if torch.cuda.is_available():
        label = label.cuda()
    print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
        self.trlog['max_acc_epoch'],
        self.trlog['max_acc'],
        self.trlog['max_acc_interval']))
    with torch.no_grad():
        for i, batch in tqdm(enumerate(self.test_loader, 1)):
            if torch.cuda.is_available():
                data = torch.stack(batch[0], 0).cuda()
                gt_label = batch[1].cuda()
            else:
                data = batch[0]

            logits, proto_loss = self.model(data)
            proto_loss = proto_loss / 80 * args.proto_ratio  # 调平
            loss = self.loss(logits, label) + proto_loss
            loss /= (1 + args.proto_ratio)  # 避免步长变大
            acc = count_acc(logits, label)
            record[i - 1, 0] = loss.item()
            record[i - 1, 1] = acc
    assert (i == record.shape[0])
    vl, _ = compute_confidence_interval(record[:, 0])
    va, vap = compute_confidence_interval(record[:, 1])

    self.trlog['test_acc'] = va
    self.trlog['test_acc_interval'] = vap
    self.trlog['test_loss'] = vl

    print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
        self.trlog['max_acc_epoch'],
        self.trlog['max_acc'],
        self.trlog['max_acc_interval']))
    print('Test acc={:.4f} + {:.4f}\n'.format(
        self.trlog['test_acc'],
        self.trlog['test_acc_interval']))

    with open(osp.join(self.args.save_path, 'cross_domain_test_CUB'),
              'w') as f:
        f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        f.write('Test acc={:.4f} + {:.4f}\n'.format(
            self.trlog['test_acc'],
            self.trlog['test_acc_interval']))
