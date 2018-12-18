import os
import sys
import copy
import argparse
import numpy as np
import pickle as pkl

import torch

from multiprocess import Pool

# import algorithms
from BBFWA import BBFWA
from basic_SRBBFWA import BasicSRBBFWA
from round_SRBBFWA import RoundSRBBFWA
from manual_SRBBFWA import ManualSRBBFWA

# import benchmark
sys.path.append("/home/lyf/Desktop/fireworks_algorithms/benchmarks")
from benchmark_wrapper import Benchmark

def parsing():
    
    # experiment setting
    parser = argparse.ArgumentParser(description='Baseline runs of dynFWA')
    parser.add_argument('--alg_name', default='BBFWA', help='Algorithm Name')
    parser.add_argument('--benchmark', default='CEC17', help='Benchmark Name')
    parser.add_argument('--multiprocess', default=False, action='store_true', help='Whether apply multiprocess.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--repetition', default=50, type=int, help='Repetition times')
    parser.add_argument('--dim', default=30, type=int, help='Dim of fitness function')

    return parser.parse_args()

def single_opt(pack):
    
    funcs, func_id, args = pack
    
    if args.alg_name == 'BBFWA':
        model = BBFWA()
    elif args.alg_name == 'basic_SRBBFWA':
        model = BasicSRBBFWA()
    elif args.alg_name == 'round_SRBBFWA':
        model = RoundSRBBFWA()
    elif args.alg_name == 'manual_SRBBFWA':
        model = ManualSRBBFWA()
    else:
        raise Exception("Algorithm not implemented!")

    model.load_prob(evaluator=funcs[func_id],
                    dim=args.dim,
                    max_eval=args.dim*10000,
                    disable_cuda=args.disable_cuda,)
    min_val, cost_time, trajectory = model.run()
    print("Prob.{:<4}, res:{:.4e},\t time:{:.3f}".format(func_id+1, min_val, cost_time))
    return min_val, cost_time, trajectory

if __name__ == "__main__":

    args = parsing()

    benchmark = Benchmark()
    
    if args.benchmark == 'CEC13':
        func_num = 28
        funcs = [benchmark.cec13[_] for _ in range(func_num)]
    elif args.benchmark == 'CEC17':
        func_num = 30
        funcs = [benchmark.cec17[_] for _ in range(func_num)]

    for i in range(func_num):
        
        dir_name = "logs/{}_{}_{}D_func_{}".format(args.alg_name, args.benchmark, args.dim, i+1)
        if not os.path.exists(dir_name):
                os.makedirs(dir_name)    
        exist_idx = [int(_.split('.')[0].split('_')[-1]) for _ in os.listdir(dir_name)]

        for j in range(args.repetition):
            if j in exist_idx:
                continue

            res, cst, traj = single_opt((funcs, i, args))
    
            with open(os.path.join(dir_name, "record_{}.pkl".format(j)), "wb") as f:
                pkl.dump({"res": res, "cst": cst, "traj": traj,}, f)
