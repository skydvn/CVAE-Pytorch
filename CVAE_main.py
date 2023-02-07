# Import Library
from utils.setting_setup import *


import torch
import numpy as np

if __name__ == "__main__":
    # get parser
    args = get_parser()

    ''' Initialization '''
    # Time parameters
    # CUDA
    # Model define
    model = CVAE(args, device)
    # get dataset



