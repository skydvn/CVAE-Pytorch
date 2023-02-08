import torch
import time

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from models.CVAE import *
from models.utils import *


def train(args):
    """ Initialization """
    # CUDA
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    dataset = MNIST(
        root='data', train=True, transform=transforms.ToTensor(),
        download=True)
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)

    # Time parameters
    start_time = last_logging = time.time()

    # Init Scaler
    scaler = torch.cuda.amp.GradScaler()

    # Loops over epochs
    for epoch in range(start_epoch, args.epochs):
        # Get data batch

        # Update learning rate

        # Calculate loss function
        with torch.cuda.amp.autocast():
            o_batch = model.forward()
            loss =

        # Update loss function
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Logging data

        pass
