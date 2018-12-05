import os
import sys
import copy
import argparse
import numpy as np
import pickle as pkl

from multiprocess import Pool

sys.path.append("/home/lyf/Desktop/FWA/algorithms")
import dynFWA

sys.path.append("/home/lyf/Desktop/FWA/benchmarks")
from benchmark_wrapper import Benchmark

def parsing():
    
    # experiment setting
    parser = argparse.ArgumentParser(description='Baseline runs of dynFWA')
    parser.add_argument('--alg_name', default='dynFWA', help='Algorithm Name')
    parser.add_argument('--benchmark', default='CEC17', help='Benchmark Name')
    parser.add_argument('--multiprocess', default=False, action='store_true', help='Whether apply multiprocess.')
    parser.add_argument('--repetition', default=50, type=int, help='Repetition times')
    parser.add_argument('--dim', default=30, type=int, help='Dim of fitness function')

    return parser.parse_args()

def single_opt(pack):
    
    funcs, func_id, args = pack

    model = dynFWA.dynFWA()
    model.load_prob(evaluator=funcs[func_id],
                    dim=args.dim,
                    max_eval=args.dim*10000)
    min_val, cost_time = model.run()
    print("Prob.{:<4}, res:{:.4e},\t time:{:.3f}".format(func_id+1, min_val, cost_time))
    return min_val, cost_time

if __name__ == "__main__":

    args = parsing()

    benchmark = Benchmark()
    
    if args.benchmark == 'CEC13':
        func_num = 28
        funcs = [benchmark.cec13[_] for _ in range(func_num)]
    elif args.benchmark == 'CEC17':
        func_num = 30
        funcs = [benchmark.cec17[_] for _ in range(func_num)]

    res = np.empty((func_num, args.repetition))
    cst = np.empty((func_num, args.repetition))

    for i in range(func_num):

        if args.multiprocess:
            p = Pool(5)
            results = p.map(single_opt, [(funcs, i, args)]*args.repetition)
            p.close()
            p.join()

            for idx, tmp in enumerate(results):
                res[i, idx] = tmp[0]
                cst[i, idx] = tmp[1]
        else:
            for j in range(args.repetition):
                res[i, j], cst[i, j] = single_opt((funcs, i, args))

    with open("logs/{}_{}_{}D.pkl".format(args.alg_name, args.benchmark, args.dim), "wb") as f:
        pkl.dump({"res": res, "cst": cst}, f)
