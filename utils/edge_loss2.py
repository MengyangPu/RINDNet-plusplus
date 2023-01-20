import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import dice

def clip_by_value(t, t_min, t_max):
    result = (t >= t_min)* t + (t < t_min) * t_min
    result = (result <= t_max) * result + (result > t_max)* t_max
    return result

def attention_loss2(output,target):
    num_pos = torch.sum(target == 1).float()
    num_neg = torch.sum(target == 0).float()
    alpha = num_neg / (num_pos + num_neg) * 1.0
    eps = 1e-14
    p_clip = torch.clamp(output, min=eps, max=1.0 - eps)

    weight = target * alpha * (4 ** ((1.0 - p_clip) ** 0.5)) + \
             (1.0 - target) * (1.0 - alpha) * (4 ** (p_clip ** 0.5))
    weight=weight.detach()

    loss = F.binary_cross_entropy(output, target, weight, reduction='none')
    loss = torch.sum(loss)
    return loss


class AttentionLoss2(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()
        total_loss = 0
        for i in range(len(output)):
            #o = output[i]
            o = output[:,i:i+1,:,:]
            l = label[:,i:i+1,:,:]
            loss_focal = attention_loss2(o, l)
            total_loss = total_loss + loss_focal
        total_loss = total_loss / batch_size
        return total_loss


class AttentionLoss2_weight(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLoss2_weight, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()

        loss_d = attention_loss2(output[:,0:1,:,:], label[:,0:1,:,:])
        loss_n = attention_loss2(output[:,1:2,:,:], label[:,1:2,:,:])
        loss_r = attention_loss2(output[:,2:3,:,:], label[:,2:3,:,:])
        loss_i = attention_loss2(output[:,3:4,:,:], label[:,3:4,:,:])

        num_d = torch.sum(label[:, 0:1, :, :] == 1).float() + 1
        num_n = torch.sum(label[:, 1:2, :, :] == 1).float() + 1
        num_r = torch.sum(label[:, 2:3, :, :] == 1).float() + 1
        num_i = torch.sum(label[:, 3:4, :, :] == 1).float() + 1

        w_depth = num_i / num_d
        w_normal = num_i / num_n
        w_reflectance = num_i / num_r
        w = w_depth + w_normal + w_reflectance + 1.0
        w_depth = w_depth / w
        w_normal = w_normal / w
        w_reflectance = w_reflectance / w
        w_illumination = 1.0 / w

        total_loss = w_depth * loss_d + w_normal * loss_n + w_reflectance * loss_r + w_illumination * loss_i
        total_loss = total_loss / batch_size
        return total_loss

class AttentionLossList(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLossList, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()
        total_loss = 0
        for i in range(len(output)):
            o = output[i]
            loss = attention_loss2(o, label)
            total_loss = total_loss + loss
        total_loss = total_loss / batch_size
        return total_loss


class AttentionLossSingleMap(nn.Module):
    def __init__(self,alpha=0.1,gamma=2,lamda=0.5):
        super(AttentionLossSingleMap, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lamda = lamda

    def forward(self,output,label):
        batch_size, c, height, width = label.size()
        loss_focal = attention_loss2(output, label)
        total_loss = loss_focal / batch_size
        return total_loss


if __name__ == '__main__':
    N = 4
    H, W = 320, 320
    label = torch.randint(0, 2, size=(N, 4, H, W)).float()
    o_b = torch.rand(N, 4, H, W)
    #o_b = [torch.rand(N, 1, H, W), torch.rand(N, 1, H, W), torch.rand(N, 1, H, W), torch.rand(N, 1, H, W)]
    #crientation = AttentionLossList()
    #total_loss = crientation(o_b, label)

    crientation = AttentionLoss2_weight()
    total_loss = crientation(o_b, label)
    print('loss 2-1 :   '+ str(total_loss))





