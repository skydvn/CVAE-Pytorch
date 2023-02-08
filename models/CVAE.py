import torch
from torch import nn
from models.base_model import *


class CVAE(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.encoder_layer_sizes = None
        self.decoder_layer_sizes = None
        self.initial_setup(args)

        if args.encoder == "dense":
            self.encoder = Encoder()
            self.decoder = Decoder()
        elif args.encoder == "conv":
            pass
        elif args.encoder == "resnet":
            pass
        else:
            pass

    def initial_setup(self, args):
        self.encoder_layer_sizes = args.en_size
        self.decoder_layer_sizes = args.de_size

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(-1, 28 * 28)

        z = self.encoder(x)
        recon_x = self.decoder(z)

        return recon_x, z
