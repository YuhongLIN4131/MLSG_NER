import torch
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target, mask=None):
        predict = predict.view(-1)
        target = target.view(-1)
        if mask is not None:
            mask = mask.view(-1).float()
            predict = predict * mask
            target = target * mask
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        #加两个概率  w0：用于假阳性的权重 为1，w1:假阴性的权重为10
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss