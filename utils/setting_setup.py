import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    # run--config
    parser.add_argument("--seed", type=int, default=730, help="one manual random seed")
    parser.add_argument("--n_seed", type=int, default=1, help="number of runs")
    parser.add_argument("--exp_dir", type=str, default=None, help="checkpoint directory")

    # --------------------- flag & name

    # --------------------- model config
    parser.add_argument("--encoder", type=str, default="dense", help="model type CVAE")
    parser.add_argument("--en_size", type=int, default=5, help="encoder layer size")
    parser.add_argument("--de_size", type=int, default=5, help="decoder layer size")

    # --------------------- train config
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--lr", type=int, default=0.005, help="learning rate")
    parser.add_argument("--wd", type=int, default=0.001, help="weight decay")

    # --------------------- loss config

    return parser.parse_args()
