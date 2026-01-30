import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import ot


def _one_hot(tensor, num):
    b = list(tensor.size())[0]
    onehot = torch.cuda.FloatTensor(b, num).fill_(0)
    ones = torch.cuda.FloatTensor(b, num).fill_(1)
    out = onehot.scatter_(1, torch.unsqueeze(tensor, 1), ones)
    return out


class L1regularization(nn.Module):
    def __init__(self, weight_decay=0.1):
        super(L1regularization, self).__init__()
        self.weight_decay = weight_decay

    def forward(self, model):
        regularization_loss = 0.
        for param in model.parameters():
            regularization_loss += torch.mean(abs(param)) * self.weight_decay

        return regularization_loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes=20, feat_dim=64, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, embeddings, labels):
        center_loss = 0
        for i, x in enumerate(embeddings):
            label = labels[i].long()
            batch_size = x.size(0)
            distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                      torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
            distmat.addmm_(1, -2, x, self.centers.t())
            distmat = torch.sqrt(distmat)

            classes = torch.arange(self.num_classes).long()
            if self.use_gpu: classes = classes.cuda()
            label = label.unsqueeze(1).expand(batch_size, self.num_classes)
            mask = label.eq(classes.expand(batch_size, self.num_classes))

            dist = distmat * mask.float()
            center_loss += torch.mean(dist.clamp(min=1e-12, max=1e+12))
        
        center_loss = center_loss/len(embeddings)
        return center_loss


class CellLoss(nn.Module):
    def __init__(self):
        super(CellLoss, self).__init__()

    def forward(self, rna_cell_out, rna_cell_label):
        rna_cell_loss = F.cross_entropy(rna_cell_out, rna_cell_label.long())
        return rna_cell_loss
