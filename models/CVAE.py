import torch
from torch import nn
from models.base_model import *


class CVAE(nn.Module):
    def __init__(self, args, device):
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
