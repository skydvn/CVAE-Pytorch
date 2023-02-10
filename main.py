# Import Library
from test_manager import *
from train_manager import *
from utils.setting_setup import *
import torch


if __name__ == "__main__":
    # get parser
    args = get_arguments()
    print(torch.cuda.is_available())

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        pass
