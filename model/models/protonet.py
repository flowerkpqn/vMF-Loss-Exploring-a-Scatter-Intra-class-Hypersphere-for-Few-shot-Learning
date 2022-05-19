import torch
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel

import torch

from .metircs import distanceMetric,spectralMetric

from .loss import kCosLogit

# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

class ProtoNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        self.k_cos_loss=kCosLogit(args.kappa, args.way*args.query,args.way*args.shot)
        if args.learn_temperature:
            self.scale_cls = torch.nn.Parameter(
                torch.FloatTensor(1).fill_(args.temperature), requires_grad=True
            )
        else:
            self.scale_cls = args.temperature
        # self.task_num=0
        self.intra_ress=[]
        self.inter_ress=[]
        self.intra_over_inter_ress=[]
        self.rho_spec_ress=[]
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

    def prototype_loss(self, prototypes):
        #normailisation
        prototypes = F.normalize(prototypes, p=2, dim=1)
        # Dot product of normalized prototypes is cosine similarity.
        product = torch.matmul(prototypes, prototypes.t()) + 1
        # Remove diagnoxnal from loss.
        product -= 2. * torch.diag(torch.diag(product))
        # Minimize maximum cosine similarity.
        loss = product.max(dim=1)[0]
        return loss.mean()

    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)
        # if self.training:
        #     self.task_num +=1

        #record->最后一轮开始统计
        # if self.training and self.task_num > (self.args.max_epoch - 1) * self.args.episodes_per_epoch:
        if self.training:
            label=[]
            for i in range(self.args.shot+self.args.query):
                for j in range(self.args.way):
                    label.append(j)
            label=torch.tensor(label,dtype=torch.int)
            feature = instance_embs.detach()
            metric = distanceMetric(mode='intra')
            intra_res = metric(feature, label)
            # print('intra_', intra_res)
            self.intra_ress.append(intra_res)
            metric = distanceMetric(mode='inter')
            inter_res = metric(feature, label)
            # print('inter_', inter_res)
            self.inter_ress.append(inter_res)
            metric = distanceMetric(mode='intra_over_inter')
            intra_over_inter_res = metric(feature, label)
            # print('intra_over_inter_', intra_over_inter_res)
            # U, s, V = torch.svd(torch.Tensor(feature))
            self.intra_over_inter_ress.append(intra_over_inter_res)
            # print('svd_',torch.sum(s))
            metric = spectralMetric(640, 1)
            rho_spec_res = metric(feature)
            # print(metric.name, ":", rho_spec_res)
            self.rho_spec_ress.append((rho_spec_res))

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))

        # get mean of the
        proto = support.mean(dim=1) # Ntask x NK x d

        #proto_loss
        if self.args.proto_ratio!=0:
            proto_loss = self.prototype_loss(proto.squeeze())
        else:
            proto_loss=0

        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if self.args.use_euclidean: # self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1) # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
            proto = proto.contiguous().view(num_batch*num_query, num_proto, emb_dim) # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.scale_cls
        elif self.args.loss_type=='cross_entropy': # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1) # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim) # (Nbatch,  Nq*Nw, d)

            # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
            logits = torch.bmm(query, proto.permute([0,2,1])) / self.scale_cls
            logits = logits.view(-1, num_proto)
        else:
            query = query.view(-1, emb_dim)
            logits = self.k_cos_loss(query, proto.squeeze()) / self.scale_cls
        if self.training:
            return logits, proto_loss, None
        else:
            return logits, proto_loss
