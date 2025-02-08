import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import torch

# KL Divergence
# kld(map2||map1) -- map2 is gt
def KLD(map1, map2, reduction="batchmean"): # map1 has been softmaxed
    assert map1.dim() == map2.dim() == 2
    assert torch.allclose(map1.sum(dim=1), torch.ones(len(map1)).to(map1.device))
    assert torch.allclose(map2.sum(dim=1), torch.ones(len(map2)).to(map2.device))
    map1, map2 = map1/map1.sum(dim=1, keepdim=True), map2/map2.sum(dim=1, keepdim=True)
    map1 += 1e-10
    kld = F.kl_div(map1.log(), map2, reduction=reduction)
    return kld

# historgram intersection
def SIM(map1, map2):
    assert map1.dim() == map2.dim() == 2
    map1, map2 = map1/map1.sum(dim=1, keepdim=True), map2/map2.sum(dim=1, keepdim=True)
    return torch.minimum(map1, map2).sum()  / len(map1)


def NSS(pred, gt):
    assert pred.dim() == gt.dim() == 2
    pred = pred / pred.max(dim=1, keepdim=True).values
    gt = gt / gt.max(dim=1, keepdim=True).values
    std = pred.std(dim=1, keepdim=True)
    u = pred.mean(dim=1, keepdim=True)

    smap = (pred - u) / std
    fixation_map = (gt - torch.min(gt, dim=1, keepdim=True).values) / (torch.max(gt, dim=1, keepdim=True).values - torch.min(gt, dim=1, keepdim=True).values + 1e-12)
    fixation_map = (fixation_map >= 0.1)
    nss = smap * fixation_map

    nss = nss.sum(dim=1) / (fixation_map.sum(dim=1) + 1e-12)
    return nss.mean()


def KL_loss(preds, target, valid, already_softmax=False):
    if already_softmax:
        preds = torch.log(preds.reshape(len(target), -1))
    else:
        preds = F.log_softmax(preds.reshape(len(target), -1), dim=-1)
    loss = F.kl_div(preds, target.reshape(len(target), -1), reduction='none')
    loss *= valid.reshape(-1, 1,)
    if valid.sum() > 0:
        loss = loss.sum() / (valid>0).sum()
        return loss
    return valid.sum()
