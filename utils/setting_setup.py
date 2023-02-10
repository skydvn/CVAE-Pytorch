import argparse
from pathlib import Path


def get_arguments():
    parser = argparse.ArgumentParser()
    # run--config
    parser.add_argument("--seed", type=int, default=730,
                        help="one manual random seed")
    parser.add_argument("--n-seed", type=int, default=1,
                        help="number of runs")

    # --------------------- Path
    parser.add_argument("--data-dir", type=Path, default="D:/Datasets/s",
                        help="Path to the mnist dataset")
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help="Path to the experiment folder, where all logs/checkpoints will be stored")

    # --------------------- flag & name
    parser.add_argument("--mode", type=str, default="train",
                        help="experiment mode")
    parser.add_argument("--log-delay", type=float, default=2.0,
                        help="Time between two consecutive logs (in seconds)")
    parser.add_argument("--eval", type=bool, default=True,
                        help="Evaluation Trigger")

    # --------------------- model config
    parser.add_argument("--encoder", type=str, default="dense",
                        help="model type CVAE")
    parser.add_argument("--en_size", type=list, default=[784, 512, 256],
                        help="encoder layer size")
    parser.add_argument("--de_size", type=list, default=[256, 512, 784],
                        help="decoder layer size")
    parser.add_argument("--latent_size", type=int, default=128,
                        help="Embedding vector size")
    # --------------------- train config
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-6,
                        help="weight decay")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Data batch size")
    parser.add_argument("--scaler", type=bool, default=False,
                        help="Trigger the torch scaler function")

    # --------------------- loss config

    return parser.parse_args()
