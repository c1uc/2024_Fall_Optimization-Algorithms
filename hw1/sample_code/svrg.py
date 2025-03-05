#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
535526 Optimization Algorithms: HW1

This file implements the stochastic variance reduced gradient (SVRG) step.

Specifically, we need to define a class (inherited from "Optimizer" in pytorch)\
that specifies the one-step update under SVRG

Reference: Pytorch official implementation of "Optimizer" at 
https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer
"""

from torch.optim import Optimizer
import copy


class SVRG(Optimizer):
    """
        This class is for calculating the gradient of one iteration.
        
        - params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        - lr (float): learning rate
    """
    def __init__(self, params, lr):
        print("Using SVRG optimizer ...")
        self.mu = None # The variable for storing the snapshot
        defaults = dict(lr=lr)
        super(SVRG, self).__init__(params, defaults)
    
    def get_param_groups(self):
            return self.param_groups

    def set_mu(self, new_mu):
        """
            Set the mean gradient for the current iteration. 
        """
        if self.mu is None:
            self.mu = copy.deepcopy(new_mu)
        for u_group, new_group in zip(self.mu, new_mu):  
            for mu, new_mu in zip(u_group['params'], new_group['params']):
                mu.grad = new_mu.grad.clone()

    def step(self, params):
        """
            Performs a single optimization step via SVRG.
            (Hint: This part is similar to that in SGD)
        """
        #---------- You Code (~10 lines) ----------

        for group, snap_group, mu_group in zip(self.param_groups, params, self.mu):
            for p, snap, mu in zip(group['params'], snap_group['params'], mu_group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                d_s = snap.grad.data
                d_mu = mu.grad.data

                p.data.add_(d_p - d_s + d_mu, alpha=-group['lr'])

        #---------- End of You Code ----------
        
class SVRG_Snapshot(Optimizer):
    """
        This class for calculating the average gradient (i.e., snapshot) of all samples.

        - params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        - lr (float): learning rate
    """
    def __init__(self, params):
        defaults = dict()
        super(SVRG_Snapshot, self).__init__(params, defaults)
      
    def get_param_groups(self):
            return self.param_groups
    
    def set_param_groups(self, new_params):
        """
            Copies the parameters from the inner-loop optimizer. 
            There are two options:
            1. Use the latest parameters
            2. Draw uniformly at random from the previous m parameters
        """
        for group, new_group in zip(self.param_groups, new_params): 
            for p, q in zip(group['params'], new_group['params']):
                  p.data[:] = q.data[:]