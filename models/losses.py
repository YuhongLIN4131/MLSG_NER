
from fastNLP import LossBase
import torch.nn.functional as F
from fastNLP import seq_len_to_mask
import torch
import torch.nn as nn
from torch.nn.modules import  BCEWithLogitsLoss
import numpy as np
import os
import random
from models.focal_loss import FocalLoss
from models.adaptive_dice_loss import AdaptiveDiceLoss
# seed_num = 666
# random.seed(seed_num)
# os.environ['PYTHONHASHSEED'] = str(seed_num)
# np.random.seed(seed_num)
# torch.manual_seed(seed_num)
# torch.cuda.manual_seed(seed_num)
# torch.cuda.manual_seed_all(seed_num)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
ASL_neg={"Onto":0,"conll03":0,"ace2004":0,"ace2005":0,'genia':0,'cadec':0,'share2013':0,'share2014':0}
ASL_pos={"Onto":0,"conll03":0,"ace2004":0,"ace2005":1.5,'genia':0,'cadec':0,'share2013':1,'share2014':1}
class Seq2SeqLoss(LossBase):
    def __init__(self,loss_type='cross_entropy',dataset_name=None):
        super().__init__()
        self.bce_loss = BCEWithLogitsLoss(reduction="mean")
        # self.loss_type=loss_type
        self.focal_loss = FocalLoss(reduction="sum")
        self.AdaptiveLoss = AdaptiveDiceLoss(reduction="sum")
        self.loss_type = "ASL"
        # self.dice_loss = DiceLoss(with_logits=True, smooth=args.dice_smooth)
        self.AsymmetricLossOptimized = AsymmetricLossOptimized(gamma_neg=ASL_neg[dataset_name],gamma_pos=ASL_pos[dataset_name])

    def get_loss(self, pred, labels = None, cal_label_mask = None):
        """
        :param tgt_tokens: bsz x max_len, 包含了的[sos, token, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        panduan 判断样本是否有有解
        """
        number = len(pred)
        loss = self.cal_loss(pred, labels,cal_label_mask)
        loss = loss / number
        # print(loss)
        return loss

    def cal_loss(self, pred, labels,cal_label_mask):
        '''选择那些token能够计算损失'''
        '''使用multilabel_categorical_crossentropy损失'''
        loss = self.multilabel_categorical_crossentropy(labels, pred, cal_label_mask).sum()
        # 测试focal_loss 和 diceloss
        # cal_pos = torch.nonzero(cal_label_mask.view(-1)).squeeze(-1)
        # loss = self.AsymmetricLossOptimized(pred.view(-1)[cal_pos],labels.view(-1)[cal_pos]).sum()
        # loss = self.focal_loss(pred.view(-1)[cal_pos],labels.view(-1)[cal_pos])
        # loss = self.AdaptiveLoss(pred.view(-1)[cal_pos],labels.view(-1)[cal_pos])
        # all_loss = all_loss+loss.sum()
        return loss
    def multilabel_categorical_crossentropy(self, y_true, y_pred,mask=None):
        """多标签分类的交叉熵
        说明：y_true和y_pred的shape一致，y_true的元素非0即1，
             1表示对应的类为目标类，0表示对应的类为非目标类。
        警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
             不用加激活函数，尤其是不能加sigmoid或者softmax！预测
             阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
             本文。
        """
        y_pred = (1 - 2 * y_true) * y_pred #全部变为了负数值
        y_pred_neg = y_pred - y_true * 2e30#把正值的分数填充为-无穷，忽略它的损失
        y_pred_pos = y_pred - (1 - y_true) * 2e30#把负值的分数填充为-无穷，忽略它的损失
        zeros = torch.zeros_like(y_pred[..., :1])
        ones = torch.ones_like(y_pred[..., :1])
        y_pred_neg = torch.cat((y_pred_neg, zeros), -1)#增加偏置项
        y_pred_pos = torch.cat((y_pred_pos, zeros), -1)
        '''使用mask矩阵去除那些填充的计算损失的点'''
        # mask = torch.cat((mask, ones), -1)#损失项给出那些需要计算损失
        # neg_loss = torch.log((torch.exp(y_pred_neg)*mask).sum(dim=-1))#也就是说实际的类别总数对于结果是有影响的
        # pos_loss = torch.log((torch.exp(y_pred_pos)*mask).sum(dim=-1))
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)#
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss
# class FocalLoss(nn.Module):
#     r"""
#     """
#     def __init__(self, alpha=0.25, gamma=2, reduction = "none"):
#         super(FocalLoss, self).__init__()
#         self.alpha_num = alpha##alpha莫仍1
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward2(self, inputs, targets):
#         '''
#         :param inputs: batch_size,class_number
#         :param targets: batch_size
#         :return:
#         '''
#         # import pdb; pdb.set_trace()
#         class_num = inputs.size(-1)
#         alpha = torch.ones(class_num, 1) * self.alpha_num
#
#         P = F.softmax(inputs, dim=-1)
#         LogSoftmax = nn.LogSoftmax(dim=1)
#         log_probs = LogSoftmax(inputs)
#         class_mask = targets
#         alpha = alpha.to(device=inputs.device)
#         alpha = alpha[ids.data.view(-1)]
#
#         probs = (P * class_mask).sum(dim=1).view(-1, 1)
#         log_probs = (log_probs * class_mask).sum(dim=1).view(-1, 1)
#         log_p = log_probs
#
#         batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
#         loss = batch_loss
#
#         if self.reduction == "mean":
#             loss = batch_loss.mean()
#         if self.reduction == "sum":
#             loss = batch_loss.sum()
#
#         return loss
#
#     def forward(self, inputs, targets, mask=None, mask2=None):
#         '''
#
#         :param inputs: batch_size,class_number
#         :param targets: batch_size
#         :return:
#         '''
#         # import pdb; pdb.set_trace()
#         class_num = inputs.size(-1)
#         alpha = torch.ones(class_num, 1) * self.alpha_num
#         N = inputs.size(0)  #
#         C = inputs.size(1)
#
#         P = F.softmax(inputs, dim=-1)
#         LogSoftmax = nn.LogSoftmax(dim=1)
#         log_probs = LogSoftmax(inputs)
#
#         class_mask = inputs.data.new(N, C).fill_(0).to(device=inputs.device)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)  # 填充one-hot
#         class_mask.masked_fill_(mask, 0)
#         alpha = alpha.to(device=inputs.device)
#         alpha = alpha[ids.data.view(-1)]
#         probs = (P * class_mask).sum(dim=1).view(-1, 1)
#         log_probs = (log_probs * class_mask).sum(dim=1).view(-1, 1)#yig
#         positions = torch.nonzero(mask2).squeeze(-1)
#         probs = probs[positions]
#         alpha = alpha[positions]
#         log_p = log_probs[positions]
#
#         batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
#         loss = batch_loss
#
#         if self.reduction == "mean":
#             loss = batch_loss.mean()
#         if self.reduction == "sum":
#             loss = batch_loss.sum()
#
#         return loss
# class LabelSmoothingLoss(nn.Module):
#     """
#     With label smoothing,
#     KL-divergence between q_{smoothed ground truth prob.}(w)
#     and p_{prob. computed by model}(w) is minimized.
#     """
#     def __init__(self, label_smoothing, ignore_index=-100):
#         assert 0.0 < label_smoothing <= 1.0
#         self.ignore_index = ignore_index
#         self.label_smoothing=label_smoothing
#         super(LabelSmoothingLoss, self).__init__()
#
#     def forward(self, output, target, mask=None,reduction="none"):
#         """
#         output (FloatTensor): batch_size x n_classes
#         target (LongTensor): batch_size
#         """
#         output = F.log_softmax(output,dim=-1)
#
#         tgt_vocab_size = output.size(-1)#标签的个数
#         smoothing_value = self.label_smoothing / tgt_vocab_size
#         one_hot = torch.full((tgt_vocab_size,), smoothing_value).to(target.device)
#         # one_hot[self.ignore_index] = 0#忽略某些类别
#         one_hot = one_hot.unsqueeze(0)
#         model_prob = one_hot.repeat(target.size(0), 1)
#         confidence = 1.0 - self.label_smoothing
#         model_prob.scatter_(1, target.unsqueeze(1), confidence)#忽略某些类别
#
#         # model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
#         model_prob.masked_fill_(mask, 0)
#
#
#         return -model_prob*output

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations
    gamma_neg越大 召回会更好，但是精度却更低
    gamma_pos越大 精度会更好些
    '''

    def __init__(self, gamma_neg=0, gamma_pos=0, clip=0, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        #给非填充值进行平滑标签
        # real_pos = torch.nonzero(x>-1e25).squeeze(-1).long()#非填充值
        if ((x<-1e25) & (y>0)).sum()>0:
            print("逻辑出错1")
            exit(0)
        if ((x>1e25) & (y<1)).sum()>0:
            print("逻辑出错2")
            exit(0)
        self.targets = y.float()
        self.anti_targets = 1 - y.float()
        # self.targets[real_pos] = self.targets[real_pos]*0.99 + (self.targets[real_pos]==0).long()*0.01
        # self.anti_targets[real_pos] = self.anti_targets[real_pos]*0.99 + (self.anti_targets[real_pos]==0).long()*0.01
        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:#去除非常容易区分的负样本
            self.xs_neg.add_(self.clip).clamp_(max=1)
        #减少离群点对损失的影响


        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w
        return -self.loss
