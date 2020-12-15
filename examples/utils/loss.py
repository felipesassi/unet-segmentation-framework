import torch
import torch.nn as nn
from torch.nn import functional as F

def soft_dice_loss(y_true, y_predicted):
    y_true = y_true.clone()
    y_predicted = y_predicted.clone()
    dim_1 = y_true.shape[0]
    y_true = y_true.contiguous().view(-1)
    y_predicted = y_predicted.contiguous().view(-1)
    mult = 2*torch.sum(y_true*y_predicted)
    sum_1 = torch.sum(torch.pow(y_true, 2))
    sum_2 = torch.sum(torch.pow(y_predicted, 2))
    dice = (mult + 1)/(sum_1 + sum_2 + 1)    
    return 1 - dice

if __name__ == "__main__":
    pass