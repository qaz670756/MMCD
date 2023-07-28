import os
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description="Hyperparameters", add_help=True)
    parser.add_argument('--clean_dir', type=str, help='YAML Config name', 
                        dest='clean_dir', default='MTBIT')


    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    dirs = os.listdir(f'results/{args.clean_dir}/lightning_logs/')
    