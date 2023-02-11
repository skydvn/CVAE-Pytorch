import torch
import time
import json
import os
import sys
import math
from statistics import mean
from pathlib import Path

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
    chk_file = Path((args.data_dir / "raw/t10k-labels-idx1-ubyte.gz"))
    if os.path.isfile(chk_file):
        print("1")
        f_down = False
    else:
        f_down = True
    print(f_down)
    dataset = MNIST(
        root=args.data_dir, train=True, transform=transforms.ToTensor(),
        download=f_down)
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)
    test_set = MNIST(
        root=args.data_dir, train=False, transform=transforms.ToTensor(),
        download=f_down)
    test_loader = DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False)
    # Time parameters
    start_time = last_logging = time.time()
    current_time = start_time
    # Init Scaler
    if args.scaler:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    """ Training phase """
    # Loops over epochs
    for epoch in range(start_epoch, args.epochs):
        train_loss_list = []
        for iteration, (x, _) in enumerate(data_loader):
            # Get data batch
            x_batch = x.to(device)

            # Update learning rate
            lr = args.lr

            # Update loss function
            if args.scaler:
                with torch.cuda.amp.autocast():
                    recon_x, embed_z = model(x_batch)
                    loss = loss_fn(recon_x, x_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                recon_x, embed_z = model(x_batch)
                loss = loss_fn(recon_x, x_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Log data when delay time meets "log_delay"
            current_time = time.time()
            train_delay = current_time - last_logging
            train_loss_list.append(loss.item())
            if (train_delay > args.log_delay) & args.log_flag:
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
        # Evaluate results
        if args.eval:
            eval_list = []
            for step, (x, _) in enumerate(test_loader):
                x_test = x.to("cpu")
                with torch.no_grad():
                    recon_x, embed_z = model(x_test)
                    eval_val = eval_fn(recon_x, x_test)
                    eval_list.append(eval_val)
            eval_val = mean(eval_list)
            print(f"epoch {epoch}: train:{mean(train_loss_list)} - eval:{eval_val} - delay: {current_time - last_logging}")
            last_logging = current_time
