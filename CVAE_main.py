# Import Library
from utils.setting_setup import *
from models.CVAE import *

import torch
import numpy as np

if __name__ == "__main__":
    # get parser
    args = get_arguments()

    ''' Initialization '''
    # CUDA
    device =
    # Model define
    model = CVAE(args, device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd,
                                 amsgrad=1)

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

    # Time parameters

    # Init Scaler

    # Loops over epochs
    for epoch in range(start_epoch, args.epochs)

