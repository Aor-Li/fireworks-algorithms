import os
import sys
import numpy as np
import torch


sys.path.append("/home/lyf/Desktop/FWA/benchmarks/cec2013")
import cec13
sys.path.append("/home/lyf/Desktop/FWA/benchmarks/cec2017")
import cec17


def func_wrapper(func, func_id):

    def wrapped(x):
        
        origin_type = type(x)
        if origin_type is not list:
            origin_shape = x.shape
            dim = origin_shape[-1]
        
            if origin_type is torch.Tensor:
                x = x.cpu().numpy()
            x = x.reshape((-1, dim)).tolist()

        if func == "cec13":
            tmp = cec13.eval(x, func_id+1)
        elif func == "cec17":
            tmp = cec17.eval(x, func_id+1)
        else:
            raise Exception("No such benchmark!")
        
        if origin_type is np.ndarray:
            return np.array(tmp).reshape(origin_shape[:-1])
        elif origin_type is torch.Tensor:
            return torch.tensor(tmp).reshape(origin_shape[:-1])
        elif type(x) is list and type(x[0]) is list:
            return tmp
        else:
            return tmp[0]

    return wrapped

class Benchmark(object):
    
    def __init__(self):
        self.cec13 = [func_wrapper("cec13", func_id) for func_id in range(28)]
        self.cec17 = [func_wrapper("cec17", func_id) for func_id in range(30)]

