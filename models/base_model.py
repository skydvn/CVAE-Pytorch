import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(self, x):
        return x
