import torch
import torch.nn as nn

# GAN
# G loss
def GAN_G_loss(D_fake, original=False):
    if original:
        return torch.mean(torch.log(D_fake))
    return torch.mean(torch.log(1-D_fake))
# D loss
def GAN_D_loss(D_real, D_fake, original=False):
    if original:
        return torch.mean(torch.log(D_real)) + torch.mean(torch.log(1-D_fake))
    return torch.mean(torch.log(1-D_real)) + torch.mean(torch.log(D_fake))


# LSGAN
# G loss
def LSGAN_G_loss(D_fake, original=False):
    if original:
        return torch.mean((D_fake-1)**2)
    return torch.mean((D_fake)**2)
# D loss
def LSGAN_D_loss(D_real, D_fake, original=False):
    if original:
        return torch.mean((D_real-1)**2) + torch.mean((D_fake)**2)
    return torch.mean((D_real)**2) + torch.mean((D_fake-1)**2)


#  WGAN
# G loss
def WGAN_G_loss(D_fake, original=False):
    if original:
        return -torch.mean(D_fake)
    return torch.mean(D_fake)
# D loss
def WGAN_D_loss(D_real, D_fake, original=False):
    if original:
        return -(torch.mean(D_real) - torch.mean(D_fake))
    return torch.mean(D_real) - torch.mean(D_fake)


# Classification

def CEloss(pred, label):
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn_2 = nn.MSELoss()
    loss = loss_fn(pred, label) # + loss_fn_2(pred, label_CM)
    return loss



def BCEloss(pred, label):
    
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn_2 = nn.MSELoss()
    label = torch.unsqueeze(label, 1).float()
    
    loss_normal = loss_fn(pred, label) # + loss_fn_2(pred, label)
    
    return loss_normal


# Metrics

def get_results(preds, labels, threshold=0.5):
    
    preds = [int(pred>=threshold) for pred in preds]
    
    ε  = 1e-8
    TP = [(pred+label) for (pred, label) in zip(preds, labels)].count(2)
    TN = [(pred+label) for (pred, label) in zip(preds, labels)].count(0)
    FP = [(pred-label) for (pred, label) in zip(preds, labels)].count(1)
    FN = [(pred-label) for (pred, label) in zip(preds, labels)].count(-1)

    acc = round((TP+TN) / (TP+TN+FP+FN+ε), 4)
    sen = round((TP)    / (TP+FN+ε), 4)
    spc = round((TN)    / (TN+FP+ε), 4)
    ydn = round(sen+spc-1, 4)
    
    return acc, sen, spc, ydn
