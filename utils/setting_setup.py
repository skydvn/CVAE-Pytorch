import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    # run--config
    parser.add_argument("--seed", type=int, default=730, help="one manual random seed")
    parser.add_argument("--n_seed", type=int, default=1, help="number of runs")

    # --------------------- flag & name

    # --------------------- train config
    parser.add_argument("--encoder", type=str, default="dense", help="model type CVAE")

    # --------------------- loss config

    return parser.parse_args()
