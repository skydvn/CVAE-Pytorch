import torch
import time
import json
import os
import sys
import math

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from models.CVAE import *
from models.loss import *
from models.optimizer import *

from collections import defaultdict


def train(args):
    """ Initialization """
    # CUDA
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Model define
    model = CVAE(args, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd,
                                 amsgrad=True)

    # Load checkpoint
    if (args.exp_dir / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    # get dataset
    if (args.data_dir / "raw/t10k-labels-idx1-ubyte.gz").is_file:
        f_down = False
    else:
        f_down = True
    dataset = MNIST(
        root=args.data_dir, train=True, transform=transforms.ToTensor(),
        download=f_down)
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)

    # Time parameters
    start_time = last_logging = time.time()

    # Init Scaler
    scaler = torch.cuda.amp.GradScaler()
    """ Training phase """
    # Loops over epochs
    for epoch in range(start_epoch, args.epochs):
        for iteration, (x, _) in enumerate(data_loader):
            # Get data batch
            x_batch = x.to(device)

            # Update learning rate
            lr = args.lr

            # Calculate loss function
            with torch.cuda.amp.autocast():
                recon_x, embed_z = model(x_batch)
                loss = loss_fn(recon_x, x_batch)

            # Update loss function
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Log data when delay time meets "log_delay"
            current_time = time.time()
            train_delay = current_time-last_logging
            if train_delay>args.log_delay:
                # print(f"iter: {iteration}, delay: {train_delay}")
                stats = dict(
                    epoch=epoch,
                    step=iteration,
                    loss=loss.item(),
                    time=int(train_delay),
                    lr=lr,
                )
                print(json.dumps(stats))
                # print(json.dumps(stats), file=stats_file)
                last_logging = current_time
