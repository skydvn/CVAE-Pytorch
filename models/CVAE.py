import torch
from torch import nn
from models.base_model import *


class CVAE(nn.Module):
    def __init__(self, args, device):
        if args.encoder == "dense":
            self.encoder = Encoder()
            self.decoder = Decoder()
        elif args.encoder == "conv":
            pass
        elif args.encoder == "resnet":
            pass
        else:
            pass