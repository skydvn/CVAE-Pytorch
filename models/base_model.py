import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size):
        super().__init__()

        self.e_network = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1],
                                                    layer_sizes[1:])):
            self.e_network.add_module(
                name=f"L{i}", module=nn.Linear(in_size, out_size))
            self.e_network.add_module(
                name=f"A{i}", module=nn.ReLU())

            self.linear_z = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x):
        x = self.e_network(x)
        z = self.linear_z(x)
        return z


class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size):
        super().__init__()

        input_size = latent_size

        self.d_network = nn.Sequential()
        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1],
                                                    layer_sizes)):
            self.d_network.add_module(
                name=f"L{i}", module=nn.Linear(in_size, out_size))
            if i + 1 < len(layer_sizes):
                self.d_network.add_module(name=f"A{i}", module=nn.ReLU())
            else:
                self.d_network.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z):
        recon_x = self.d_network(z)

        return recon_x
