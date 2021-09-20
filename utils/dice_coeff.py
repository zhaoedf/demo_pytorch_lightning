

import torch
import torch.nn as nn

class Dice_coeff(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, pred, target):
        b = pred.size(0)
        p = pred.view(b,-1)
        t = target.view(b,-1)
        
        inter = (p*t).sum(1) + self.eps
        union = p.sum(1) + t.sum(1) + self.eps
        coeff = (2*inter /union)
        # print('$'*100, coeff.shape,coeff, target.max())
        return coeff