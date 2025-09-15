# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F
#from configs import config

class Pidnet_loss(nn.Module):
  def __init__(self,config, weight = None):
    super(Pidnet_loss,self).__init__()
    self.config = config
    self.ohem = config.loss.use_ohem
    self.bd_loss = BondaryLoss()
    if self.ohem:
      self.sem_loss = OhemCrossEntropy(config = config,
                                        ignore_label=self.config.train.ignore_label,
                                        thres=self.config.loss.ohemthres,
                                        min_kept=self.config.loss.ohemkeep,
                                        weight=weight)
    else:
      self.sem_loss = CrossEntropy(config = config,
                                  ignore_label=self.config.train.ignore_label,
                                    weight=weight)


  def forward(self,score,labels,bd_gt):
    h, w = labels.size(1), labels.size(2)
    ph, pw = score[0].size(2), score[0].size(3)
    if ph != h or pw != w:
        for i in range(len(score)):
            score[i] = F.interpolate(score[i], size=(
                h, w), mode='bilinear', align_corners=self.config.model.align_corners)

    # for i in score[:-1]:
    #   print(i.shape)
    # print(f'labels:{labels.shape}')

    loss_s = self.sem_loss(score[:-1], labels)
    # print(f'loss_s:{loss_s}')

    loss_b = self.bd_loss(score[-1], bd_gt)

    filler = torch.ones_like(labels) * self.config.train.ignore_label
    bd_label = torch.where(F.sigmoid(score[-1][:,0,:,:])>0.8, labels, filler)
    loss_sb = self.sem_loss(score[-2], bd_label)
    loss = loss_b + loss_s + loss_sb
    ## losses
    # loss_b: 20*l_1
    # loss_s: 0.4l_0 + l_2
    # loss_sb: l_3 
    return loss

    
    

class CrossEntropy(nn.Module):
    def __init__(self, config,ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.config = config
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if self.config.model.num_outputs == 1:
            score = [score]

        balance_weights = self.config.loss.balance_weights
        sb_weights = self.config.loss.sb_weights
        if len(balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
        elif len(score) == 1:
            return sb_weights * self._forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")

        


class OhemCrossEntropy(nn.Module):
    def __init__(self, config,ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.config = config
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):


        loss = self.criterion(score, target)

        return loss.mean()

    def _ohem_forward(self, score, target, **kwargs):

        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        balance_weights = self.config.loss.balance_weights
        sb_weights = self.config.loss.sb_weights
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * \
                (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])
        
        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce
        
    def forward(self, bd_pre, bd_gt):

        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        
        return loss
    
if __name__ == '__main__':
    a = torch.zeros(2,64,64)
    a[:,5,:] = 1
    pre = torch.randn(2,1,16,16)
    
    Loss_fc = BondaryLoss()
    loss = Loss_fc(pre, a.to(torch.uint8))

        
        
        