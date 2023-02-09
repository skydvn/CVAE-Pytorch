# import torch
import torch.nn.functional as F


def loss_fn(recon_x, x):
    print(f"{recon_x.size()} - {recon_x.view(-1, 28*28).size()}")
    print(f"{x.size()} - {x.view(-1, 28*28).size()}")
    BCE = F.binary_cross_entropy(recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction="mean")
    return BCE

# def loss_fn(recon_x, x, mean, log_var):
#     BCE = torch.nn.functional.binary_cross_entropy(
#         recon_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction='sum')
#     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#
#     return (BCE + KLD) / x.size(0)
