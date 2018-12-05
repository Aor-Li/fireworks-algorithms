import os
import sys
import numpy as np


sys.path.append("/home/lyf/Desktop/FWA/benchmarks/cec2013")
import cec13
sys.path.append("/home/lyf/Desktop/FWA/benchmarks/cec2017")
import cec17


def func_wrapper(func, func_id):

    def wrapped(x):
        
        origin_shape = x.shape
        dim = origin_shape[-1]
        
        if type(x) is np.ndarray:
            x = x.reshape((-1, dim)).tolist()
        if func == "cec13":
            tmp = cec13.eval(x, func_id+1)
        elif func == "cec17":
            tmp = cec17.eval(x, func_id+1)
        else:
            raise Exception("No such benchmark!")

        return np.array(tmp).reshape(origin_shape[:-1])

    return wrapped


class Benchmark(object):
    
    def __init__(self):
        self.cec13 = [func_wrapper("cec13", func_id) for func_id in range(28)]
        self.cec17 = [func_wrapper("cec17", func_id) for func_id in range(30)]

