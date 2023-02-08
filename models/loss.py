import torch
import torch.nn.functional as F


def loss_fn(recon_x, x):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction=sum)
    return BCE / x.size(0)

# def loss_fn(recon_x, x, mean, log_var):
#     BCE = torch.nn.functional.binary_cross_entropy(
#         recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction='sum')
#     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#
#     return (BCE + KLD) / x.size(0)
