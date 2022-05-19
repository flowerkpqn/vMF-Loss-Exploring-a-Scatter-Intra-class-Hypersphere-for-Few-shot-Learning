import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)

from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm
import random
import os

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

        self.s_ = torch.nn.Parameter(torch.zeros(1)).cuda()

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        init_label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        init_label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)

        init_label = init_label.type(torch.LongTensor)
        init_label_aux = init_label_aux.type(torch.LongTensor)

        if torch.cuda.is_available():
            init_label = init_label.cuda()
            init_label_aux = init_label_aux.cuda()

        return init_label, init_label_aux

    def loss(self, Z, target):
        if self.args.loss_type=='VMF':
            s = F.softplus(self.s_).add(1.)
            l = F.cross_entropy(Z.mul(s), target, weight=None, ignore_index=-100, reduction='mean')
            return l
        else:
            return F.cross_entropy(Z, target)

    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        # start FSL training
        label, label_aux = self.prepare_label()

        #随机翻转一些query
        mask=torch.rand_like(label, dtype=float)
        mask=mask.cuda()
        mask_1=torch.rand_like(label_aux, dtype=float)
        for k in range(args.way):
            mask_1[k]=1
        mask_1=mask_1.cuda()

        # 对label进行随机翻转

        label = torch.where(mask > args.p_noise, label, torch.randint(0, args.way, (1,)).cuda())
        label_aux = torch.where(mask_1 > args.p_noise, label_aux, torch.randint(0, args.way, (1,)).cuda())

        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()

            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            start_tm = time.time()
            for batch in self.train_loader:
                self.train_step += 1

                if torch.cuda.is_available():
                    data=torch.stack(batch[0],0).cuda()
                    gt_label = batch[1].cuda()
                else:
                    data, gt_label = batch[0], batch[1]

                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # get saved centers
                logits, reg_logits, proto_loss = self.para_model(data)


                if reg_logits is not None:
                    loss = self.loss(logits, label)
                    proto_loss = proto_loss / 80 * args.proto_ratio  # 调平
                    total_loss = loss + args.balance * self.loss(reg_logits, label_aux)+proto_loss
                    total_loss /= (1 + args.proto_ratio)  # 避免步长变大
                else:
                    loss = self.loss(logits, label)
                    proto_loss=proto_loss/80*args.proto_ratio#调平
                    total_loss = self.loss(logits, label)+proto_loss
                    total_loss/=(1+args.proto_ratio)#避免步长变大

                tl2.add(loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)

                # refresh start_tm
                start_tm = time.time()

            self.lr_scheduler.step()
            self.try_evaluate(epoch)
            # print(self.para_model.scale_cls)

            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')
        len_records=len(self.para_model.intra_ress)
        # print(len_records)

        intra_ress_mean = np.mean(self.para_model.intra_ress[len_records-100:])
        inter_ress_mean = np.mean(self.para_model.inter_ress[len_records-100:])
        intra_over_inter_ress_mean = np.mean(self.para_model.intra_over_inter_ress[len_records-100:])
        rho_spec_ress_mean = np.mean(self.para_model.rho_spec_ress[len_records-100:])
        record_path=os.path.join(args.save_path, 'record_100')
        with open(record_path,'w') as f:
            f.write('intra_ress_mean  {}, inter_ress_mean ={}, \n intra_over_inter_ress_mean={}, rho_spec_ress_mean={},\n'.format(
                str(intra_ress_mean ),
                str(inter_ress_mean ),
                str(intra_over_inter_ress_mean),
                str(rho_spec_ress_mean),))

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    data=torch.stack(batch[0],0).cuda()
                    gt_label = batch[1].cuda()
                else:
                    data = batch[0]

                logits, proto_loss = self.model(data)
                proto_loss=proto_loss/80*args.proto_ratio  # 调平
                loss = self.loss(logits, label) + proto_loss
                loss /= (1+args.proto_ratio)  # 避免步长变大
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc

        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])

        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((1000, 2)) # loss and acc
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
                    data=torch.stack(batch[0],0).cuda()
                    gt_label = batch[1].cuda()
                else:
                    data = batch[0]

                logits, proto_loss = self.model(data)
                proto_loss=proto_loss/80*args.proto_ratio  # 调平
                loss = self.loss(logits, label) + proto_loss
                loss /= (1+args.proto_ratio)  # 避免步长变大
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])

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

        return vl, va, vap

    def final_record(self):
        # save the best performance in a txt file

        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))
