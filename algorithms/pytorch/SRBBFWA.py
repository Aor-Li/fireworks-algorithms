import os
import time
import numpy as np

import torch
import gpytorch
from tqdm import tqdm


EPS = 1e-8

class GPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SRBBFWA(object):

    def __init__(self):
        # Parameters
        # params of method
        self.sp_size = None     # total spark size
        self.init_amp = None    # initial dynamic amplitude

        # params of problem
        self.evaluator = None
        self.dim = None
        self.upper_bound = None

        self.max_iter = None
        self.max_eval = None

        self.likelihood = None
        self.gpr = None
        self.device = None
        self.sample_size = None
        self.gp_train_iter = None

        # States
        # private states
        self._num_iter = None
        self._num_eval = None
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
                  sp_size = 200,
                  init_amp = 200,
                  # GP
                  disable_cuda = False,
                  ):

        # load params
        self.evaluator = evaluator
        self.dim = dim
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.max_eval = max_eval
        self.max_iter = min(max_iter, int(max_eval / sp_size))

        self.sp_size = sp_size
        self.init_amp = init_amp

        self.sample_size = sp_size * 10
        self.gp_train_iter = 0
        if not disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # init states
        self._num_iter = 0
        self._num_eval = 0
        self._dyn_amp = init_amp

        self.best_idv = None
        self.best_fit = None

        self._gp_means = None
        self._gp_amp = None

        # init inspection info
        self.time = 0
        self.gp_train_time = 0
        self.info = {}

        # init random seed
        np.random.seed(int(os.getpid()*time.clock()))
        torch.manual_seed(int(os.getpid()*time.clock()))
    
    def run(self):
        begin_time = time.time()

        fireworks, fits = self._init_fireworks()
        for idx in tqdm(range(self.max_iter)):

            if self._terminate():
                break
            
            fireworks, fits = self.iter(fireworks, fits)
            #print(self.gp_train_time / (time.time() - begin_time))

        self.time = time.time() - begin_time

        return self.best_fit, self.time

    def _init_fireworks(self):

        fireworks = torch.rand((1, self.dim), device=self.device)*(self.upper_bound - self.lower_bound) + self.lower_bound
        fits = self.evaluator(fireworks).to(device=self.device)

        self._gp_mean = fireworks
        self._gp_amp = 1
            
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=self.device)
        self.gpr = GPModel(torch.zeros(size=(1, self.dim)).to(device=self.device), fits, self.likelihood).to(device=self.device)
        
        return fireworks, fits
    
    def _terminate(self):
        
        if self._num_iter >= self.max_iter:
            return True
        if self._num_eval >= self.max_eval:
            return True
        return False

    def iter(self, fireworks, fits):

        e_sparks, e_fits = self._explode(fireworks, fits)

        n_fireworks, n_fits = self._select(fireworks, fits, e_sparks, e_fits)

        # update states
        if n_fits[0] < fits[0]:
            self._dyn_amp *= 1.2
        else:
            self._dyn_amp *= 0.9

        self._num_iter += 1
        self._num_eval += len(e_sparks)

        self.best_idv = n_fireworks[0]
        self.best_fit = n_fits.cpu().numpy()[0]

        fireworks = n_fireworks
        fits = n_fits

        #print(len(e_sparks), self.best_fit, self._dyn_amp)

        return fireworks, fits

    def _explode(self, fireworks, fits):
        
        bias = (2*torch.rand([self.sample_size, self.dim], device=self.device)-1)*self._dyn_amp
        rand_samples = torch.rand((self.sample_size, self.dim), device=self.device)*(self.upper_bound - self.lower_bound) + self.lower_bound
        e_sparks = fireworks + bias
        in_bound = (e_sparks > self.lower_bound) * (e_sparks < self.upper_bound)
        in_bound = in_bound.float()
        e_sparks = in_bound * e_sparks + (1 - in_bound) * rand_samples
        
        # surogate model evaluation
        self.likelihood.eval()
        self.gpr.eval()
        with torch.no_grad():
            gp_inputs = (e_sparks - self._gp_mean) / self._gp_amp
            pred = self.gpr(gp_inputs)
            acq = pred.mean - 2*torch.sqrt(pred.variance)
        
        # selection
        idxs = torch.argsort(acq)[:self.sp_size]
        e_sparks = torch.index_select(e_sparks, 0, idxs)
        
        e_fits = self.evaluator(e_sparks).to(device=self.device)
        
        # normalization
        self._gp_mean = fireworks
        self._gp_amp = self._dyn_amp
        gp_input = (e_sparks - self._gp_mean) / self._gp_amp

        # train gpr
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=self.device)
        self.gpr = GPModel(gp_input, e_fits, self.likelihood).to(device=self.device)
        
        self.likelihood.train()
        self.gpr.train()

        optimizer = torch.optim.Adam([
            {'params': self.gpr.parameters()},
        ], lr=0.1)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gpr)
        
        t0 = time.time()
        for i in range(self.gp_train_iter):
            optimizer.zero_grad()
            output = self.gpr(gp_input)
            loss = -mll(output, e_fits)
            loss.backward()
            optimizer.step()
        self.gp_train_time += time.time() - t0
        
        return e_sparks, e_fits

    def _select(self, fireworks, fits, e_sparks, e_fits):
        idvs = torch.cat((fireworks, e_sparks))
        fits = torch.cat((fits, e_fits))
        idx = torch.argmin(fits)
        return idvs.index_select(dim=0, index=idx), fits.index_select(dim=0, index=idx)
