import argparse


def configuration():
    parser = argparse.ArgumentParser(description="Model helper data")

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--data_dir', default="./Data/BIPED/BIPED/edges", type=str)
    parser.add_argument('--save_dir', default="./Trained_Model", type=str)
    parser.add_argument('--model_state', default='train', choices=['train', 'predict'], type=str)
    args = parser.parse_args()

    return args
