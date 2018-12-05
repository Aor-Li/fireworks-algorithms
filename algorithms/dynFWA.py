import os
import time
import numpy as np

EPS = 1e-8

class dynFWA(object):

    def  __init__(self):
        # Parameters

        # params of method
        self.fw_size = None       # num of fireworks
        self.sp_size = None       # total spark size
        self.init_amp = None      # initial dynamic amplitude
        
        # params of problem
        self.evaluator = None
        self.dim = None
        self.upper_bound = None
        self.lower_bound = None

        self.max_iter = None
        self.max_eval = None

        # States

        # private states
        self._num_iter = None
        self._num_eval = None
        self._base_amp = None
        self._dyn_amp = None

        # public states
        self.best_idv = None    # best individual found
        self.best_fit = None    # best fitness found

        # for inspection
        self.time = None
        self.info = None

    def load_prob(self, 
                  # params for prob
                  evaluator = None,
                  dim = 2,
                  upper_bound = 100,
                  lower_bound = -100,
                  max_iter = 10000,
                  max_eval = 20000,
                  # params for method
                  fw_size = 5,
                  sp_size = 200,
                  base_amp = 40,
                  init_amp = 200,
                  ):

        # load params
        self.evaluator = evaluator
        self.dim = dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.max_iter = max_iter
        self.max_eval = max_eval

        self.fw_size = fw_size
        self.sp_size = sp_size
        self.init_amp = init_amp
        
        # init states
        self._num_iter = 0
        self._num_eval = 0
        self._base_amp = base_amp
        self._dyn_amp = init_amp

        self.best_idv = None
        self.best_fit = None

        # init inspection info
        self.time = 0
        self.info = {}

        # init random seed
        np.random.seed(int(os.getpid()*time.clock()))

    def run(self):
        begin_time = time.clock()

        fireworks, fits = self._init_fireworks()
        for idx in range(self.max_iter):
            
            if self._terminate():
                break

            fireworks, fits = self.iter(fireworks, fits)
        
        self.time = time.clock() - begin_time

        return self.best_fit, self.time

    def iter(self, fireworks, fits):
        
        e_sparks, e_fits = self._explode(fireworks, fits)
         
        n_fireworks, n_fits = self._select(fireworks, fits, e_sparks, e_fits)    

        # update states

        # dynamic amps
        if np.min(n_fits) < np.min(fits) - EPS:
            self._dyn_amp *= 1.2
        else:
            self._dyn_amp *= 0.9

        # iter and eval num
        self._num_iter += 1
        self._num_eval += e_sparks.shape[0]
        
        # record best results
        min_idx = np.argmin(n_fits)
        self.best_idv = n_fireworks[min_idx]
        self.best_fit = n_fits[min_idx]
        
        # new fireworks
        fireworks = n_fireworks
        fits = n_fits
        
        return fireworks, fits

    def _init_fireworks(self):

        fireworks = np.random.uniform(self.lower_bound,
                                      self.upper_bound,
                                      [self.fw_size, self.dim])
        fits = self.evaluator(fireworks)

        return fireworks, fits

    def _terminate(self):
        if self._num_iter >= self.max_iter:
            return True
        if self._num_eval >= self.max_eval:
            return True
        return False

    def _explode(self, fireworks, fits):
        
        # alocate sparks
        fits_to_max = max(fits) - np.array(fits)
        num_sparks = self.sp_size * (fits_to_max + EPS) / (sum(fits_to_max) + EPS);
        num_sparks.clip(min=1)
        num_sparks = num_sparks.astype(int).tolist()
        sum_sparks = sum(num_sparks)

        # compute amplitude
        cf_idx = np.argmin(fits)
        fits_to_min = np.array(fits) - min(fits)
        amps = self._base_amp * (fits_to_min + EPS) / (sum(fits_to_min) + EPS);
        amps[cf_idx] = self._dyn_amp
        amps = amps.tolist()

        # explode
        bias = np.random.uniform(-1, 1, [sum_sparks, self.dim])
        bias = bias * sum([[[amps[t]]]*num_sparks[t] for t in range(self.fw_size)], [])
        e_sparks = bias + sum([[fireworks[t]]*num_sparks[t] for t in range(self.fw_size)], [])

        # mapping
        rand_samples = np.random.uniform(self.lower_bound, self.upper_bound, [sum_sparks, self.dim])
        in_bound = (e_sparks > self.lower_bound) * (e_sparks < self.upper_bound)
        e_sparks = in_bound * e_sparks + (1 - in_bound) * rand_samples
        e_fits = self.evaluator(e_sparks)
        return e_sparks, e_fits

    def _select(self, fireworks, fits, e_sparks, e_fits):
        idvs = np.concatenate((fireworks, e_sparks), axis=0)
        fits = np.concatenate((fits, e_fits), axis=0)

        min_idx = np.argmin(fits)
        rand_idx = np.random.randint(0, len(idvs), self.fw_size-1)
        idx = rand_idx.tolist()
        idx.append(min_idx)

        n_fireworks = idvs[idx, :]
        n_fits = fits[idx]

        return n_fireworks, n_fits
