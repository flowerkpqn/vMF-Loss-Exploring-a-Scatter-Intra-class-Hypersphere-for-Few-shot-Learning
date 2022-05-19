#-*- coding=utf-8 -*-
#@Time:21/4/2022 下午 3:54
#@Author:kpqn
#@File:metircs.py
#@software:PyCharm
import numpy as np
import os
import torch
from scipy.spatial import distance
from scipy.stats import entropy

class distanceMetric():
    def __init__(self, mode, **kwargs):
        self.mode = mode
        self.requires = ['features', 'target_labels']
        self.name = 'dists@{}'.format(mode)

    def __call__(self, features, target_labels):
        features_locs = []
        for lab in np.unique(target_labels):
            features_locs.append(np.where(target_labels == lab)[0])

        if 'intra' in self.mode:
            if isinstance(features, torch.Tensor):
                intrafeatures = features.detach().cpu().numpy()
            else:
                intrafeatures = features

            intra_dists = []
            for loc in features_locs:
                c_dists = distance.cdist(
                    intrafeatures[loc], intrafeatures[loc], 'cosine')
                c_dists = np.sum(c_dists)/(len(c_dists)**2-len(c_dists))
                intra_dists.append(c_dists)
            intra_dists = np.array(intra_dists)
            maxval = np.max(intra_dists[1-np.isnan(intra_dists)])
            intra_dists[np.isnan(intra_dists)] = maxval
            intra_dists[np.isinf(intra_dists)] = maxval
            dist_metric = dist_metric_intra = np.mean(intra_dists)

        if 'inter' in self.mode:
            if not isinstance(features, torch.Tensor):
                coms = []
                for loc in features_locs:
                    com = normalize(
                        np.mean(features[loc], axis=0).reshape(1, -1)).reshape(-1)
                    coms.append(com)
                mean_inter_dist = distance.cdist(
                    np.array(coms), np.array(coms), 'cosine')
                dist_metric = dist_metric_inter = np.sum(
                    mean_inter_dist)/(len(mean_inter_dist)**2-len(mean_inter_dist))
            else:
                coms = []
                for loc in features_locs:
                    com = torch.nn.functional.normalize(torch.mean(
                        features[loc], dim=0).reshape(1, -1), dim=-1).reshape(1, -1)
                    coms.append(com)
                mean_inter_dist = 1 - \
                    torch.cat(coms, dim=0).mm(
                        torch.cat(coms, dim=0).T).detach().cpu().numpy()
                dist_metric = dist_metric_inter = np.sum(
                    mean_inter_dist)/(len(mean_inter_dist)**2-len(mean_inter_dist))

        if self.mode == 'intra_over_inter':
            dist_metric = dist_metric_intra / \
                np.clip(dist_metric_inter, 1e-8, None)

        return dist_metric

class spectralMetric():
    def __init__(self, embed_dim, mode,  **kwargs):
        self.mode = mode
        self.embed_dim = embed_dim
        self.requires = ['features']
        self.name = 'rho_spectrum@'+str(mode)
        self.s = None
        self.s = None

    def __call__(self, features):
        if not self.s:
            if isinstance(features, torch.Tensor):
                _, s, _ = torch.svd(features)
                s = s.cpu().numpy()
                self.s = s
            else:
                self.svd = TruncatedSVD(
                    n_components=self.embed_dim - 1, n_iter=5, random_state=42)
                self.svd.fit(features)
                self.s = self.svd.singular_values_

        if self.mode != 0:
            self.s = self.s[np.abs(self.mode)-1:]
        s_norm = self.s/np.sum(self.s)
        uniform = np.ones(len(self.s))/(len(self.s))

        if self.mode < 0:
            kl = entropy(s_norm, uniform)
        if self.mode > 0:
            kl = entropy(uniform, s_norm)
        if self.mode == 0:
            kl = s_norm

        return kl

