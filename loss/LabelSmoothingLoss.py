import torch
from torch import nn
import torch.nn.functional as F
class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        self.label_smoothing=label_smoothing
        super(LabelSmoothingLoss, self).__init__()

    def forward(self, output, target, mask=None,reduction="none"):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        output = F.log_softmax(output,dim=-1)

        tgt_vocab_size = output.size(-1)#标签的个数
        smoothing_value = self.label_smoothing / tgt_vocab_size
        one_hot = torch.full((tgt_vocab_size,), smoothing_value).to(target.device)
        # one_hot[self.ignore_index] = 0#忽略某些类别
        one_hot = one_hot.unsqueeze(0)
        model_prob = one_hot.repeat(target.size(0), 1)
        confidence = 1.0 - self.label_smoothing
        model_prob.scatter_(1, target.unsqueeze(1), confidence)#忽略某些类别

        # model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
        model_prob.masked_fill_(mask, 0)


        return -model_prob*output