import torch
import torch.nn.functional as F
import numpy as np

def loss_func(args,node,nei,rec_x,rec_adj,feature,adj,mask,MINE,MINE_prime):

    rec_loss = args.alpha*sce_loss(rec_x[mask],feature[mask],args) + (1-args.alpha)*sce_loss(rec_adj[mask],adj[mask],args)
    ###
    corr_loss = my_corr_loss(args,node,nei,rec_x,rec_adj,feature,adj,mask,MINE,MINE_prime)
    return corr_loss , rec_loss
def anomaly_score(args,z1,z2,rec_x,rec_adj,feature,adj,mask,MINE):
    z1 = z1.squeeze()
    z2 = z2.squeeze()
    rec = args.alpha*sce_score(rec_x[mask],feature[mask],args) + \
          (1-args.alpha)*sce_score(rec_adj[mask],adj[mask],args)

    scores = F.mse_loss(z1[mask], z2[mask], reduction='none').mean(1) * rec

    return scores


def sce_loss(x, y, args):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(args.gma)
    loss = loss.mean()
    return loss
def sce_score(x, y,args):#
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(args.gma)
    return loss


def my_corr_loss(args,z1,z2,rec_x,rec_adj,feature,adj,mask,MINE,MINE_prime):

    return CCA_loss(z1,z2,args)

def CCA_loss(z1,z2,args):
    z1 = z1.squeeze(1)
    z2 = z2.squeeze(1)
    c_ = torch.mm(z1.T, z2)
    c1 = torch.mm(z1.T, z1)
    c2 = torch.mm(z2.T, z2)
    c = -torch.diagonal(c_)
    loss_inv = c.mean()
    if z1.is_cuda:
        iden = torch.as_tensor(torch.eye(c.shape[0])).cuda()
    else:
        iden = torch.as_tensor(torch.eye(c.shape[0]))
    loss_dec1 = (iden - c1).pow(2).mean()
    loss_dec2 = (iden - c2).pow(2).mean()
    loss = loss_inv + args.lamda * (loss_dec1 + loss_dec2)
    return loss
def CCA(z1,z2,args):
    z1 = z1.squeeze(1)
    z2 = z2.squeeze(1)
    c_ = torch.mm(z1.T, z2)
    c1 = torch.mm(z1.T, z1)
    c2 = torch.mm(z2.T, z2)
    c = -torch.diagonal(c_)
    loss_inv = c.mean()
    if z1.is_cuda:
        iden = torch.as_tensor(torch.eye(c.shape[0])).cuda()
    else:
        iden = torch.as_tensor(torch.eye(c.shape[0]))
    loss_dec1 = (iden - c1).pow(2).mean()
    loss_dec2 = (iden - c2).pow(2).mean()
    return loss_inv,args.lamda * (loss_dec1 + loss_dec2)

