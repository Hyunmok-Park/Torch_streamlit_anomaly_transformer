import os
import argparse
import pickle

from torch.backends import cudnn
from utils.utils import *

from solver import Solver

import torch
torch.manual_seed(4)

def str2bool(v):
    return v.lower() in ('true')

def main(config):
    cudnn.benchmark = True

    # if (not os.path.exists(config.model_save_path)):
    #     mkdir(config.model_save_path)

    if config.mode == 'train':
        solver = Solver(vars(config))
        torch.autograd.set_detect_anomaly(True)
        solver.train(vars(config))
    elif config.mode == 'test':
        new_config = pickle.load(open(os.path.join(config.model_save_path, "config.p"), "rb"))

        new_config['model_save_path'] = config.model_save_path
        new_config['data_path'] = config.data_path
        new_config['win_size'] = config.win_size
        new_config['step'] = config.step

        solver = Solver(new_config)
        solver.test(new_config)
    else:
        solver = Solver(vars(config))
        torch.autograd.set_detect_anomaly(True)
        model_save_path = solver.train(vars(config))
        solver.test(vars(config), model_save_path)

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--win_size', type=int, default=50)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'all'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='None')
    parser.add_argument('--anomaly_ratio', type=float, default=4.00)
    parser.add_argument('--step', type=int, default=50)

    parser.add_argument('--dmodel', type=int, default=1024)
    parser.add_argument('--dff', type=int, default=1024)
    parser.add_argument('--elayers', type=int, default=4)
    parser.add_argument('--patience', type=int, default=3)

    parser.add_argument('--random_seed', type=int, default=4)

    config = parser.parse_args()

    args = vars(config)
    # print('------------ Options -------------')
    # for k, v in sorted(args.items()):
    #     print('%s: %s' % (str(k), str(v)))
    # print('-------------- End ----------------')
    main(config)
