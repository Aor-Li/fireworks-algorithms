import os 
import numpy as np
import pickle as pkl

import scipy.stats
from prettytable import PrettyTable


# data
# benchmark optimal


def load_result(*raw_paths):
    """
    Data loading of multiple results

    Inputs:
        *raw_paths (string): path or filename in \logs
            (filename is formatted like {}_{}_{}.pkl, last two fields defines the problem)

    Outputs:
        names   (list): data names
        res     (list of np.ndarray): optimization results
        cst     (list of np.ndarray): cost times
        ...     (adding)
    """
    paths = []
    for path in raw_paths:
        if not os.path.isabs(path):
            paths.append(os.path.join("/home/lyf/Desktop/fireworks_algorithms/logs/", path))
    
    prob = raw_paths[0].split('/')[-1]
    prob = '_'.join(prob.split('_')[-2:])
    prob = prob[:-4]
    names = ['_'.join((filename.split('/')[-1]).split('_')[:-2]) for filename in raw_paths]
    
    res = []
    cst = []
    for path in paths:
        with open(path, 'rb') as f:
            res_dict = pkl.load(f)
        res.append(res_dict['res'])
        cst.append(res_dict['cst'])
     
    return names, prob, res, cst

def stats_compare(*paths, **kwargs):
    """
    Statistically comparing of results 

    Inputs:
        *paths (string): path or filename in \logs
    """
    # handle selective kwargs
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 0.05

    # read data
    names, prob, res, cst = load_result(*paths)

    means = [_.mean(axis=1).tolist() for _ in res]
    stds = [_.std(axis=1).tolist() for _ in res]
    times = [_.mean(axis=1).tolist() for _ in cst]
    
    # benchmark num
    benchmark_num = len(means[0])

    if len(names) == 2:

        # stats analysis
        p_values = []
        signs = []
        for idx in range(benchmark_num):
            p =  scipy.stats.ranksums(res[0][idx,:], res[1][idx,:])[1]
            p_values.append(p)
            if p >= alpha:
                signs.append('=')
            else:
                if means[0][idx] < means[1][idx]:
                    signs.append('+')
                else:
                    signs.append('-')

        # prepare table
        tb = PrettyTable()
        tb.field_names = ['idx','alg1.mean','alg1.std','alg2.mean','alg2.std','P-value','Sig']
        for idx in range(benchmark_num):
            row = [str(idx+1), 
                   '{:.3e}'.format(means[0][idx]),
                   '{:.3e}'.format(stds[0][idx]),
                   '{:.3e}'.format(means[1][idx]),
                   '{:.3e}'.format(stds[1][idx]),
                   '{:.2f}'.format(p_values[idx]),
                   signs[idx],]
            if row[-1] == '+':
                for idx in range(5):
                    row[idx] = '\033[1m' + row[idx] + '\033[0m'
            tb.add_row(row)
        print("Comparing on {}: alg1: {}, alg2: {}".format(prob, names[0], names[1]))
        print(tb)
    else:

        # make pairwise stats analysis
        return

if __name__ == '__main__':
    
    res_name_1 = 'BBFWA_CEC17_30D.pkl'
    res_name_2 = 'BBFWA_torch_CEC17_30D.pkl'

    stats_compare(res_name_1, res_name_2)

