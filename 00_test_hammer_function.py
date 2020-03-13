#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:27:09 2020

@author: andrew
"""

    def _hammer_function_precompute(self):
        """
        Pre-computes the parameters of a penalizer centered at x0.
        """
        x0 = self.x_batch
        best = self.best
        surrogate = self.surrogate
        L = self.L

        assert x0 is not None

        if len(x0.shape) == 1:
            x0 = x0[None, :]
        m = surrogate.predict(x0)[0]
        pred = surrogate.predict(x0)[1].copy()
        pred[pred < 1e-16] = 1e-16
        s = np.sqrt(pred)
        r_x0 = np.abs(m - best) / L
        s_x0 = s / L
        r_x0 = r_x0.flatten()
        s_x0 = s_x0.flatten()
        return r_x0, s_x0


    def _hammer_function(self, x, x0, r, s):
        '''
        Creates the function to define the exclusion zones
        '''
        return norm.logcdf((np.sqrt( (np.square( np.atleast_2d(x)[:, None, :] - np.atleast_2d(x0)[None, :, :])).sum(-1)) - r) / s)
    
    
    #   cProfile.run('np.linalg.norm(V,axis=1)')              # 9 function calls in 0.029 seconds
    # cProfile.run('np.sqrt((V ** 2).sum(-1))')               # 5 function calls in 0.028 seconds
    
    def _penalized_acquisition(self, x):
        '''
        Creates a penalized acquisition function using 'hammer' functions
        around the points collected in the batch
        .. Note:: the penalized acquisition is always mapped to the log
        space. This way gradients can be computed additively and are more
        stable.
        '''
        fval = self.acq.evaluate(x)
        x_batch = self.x_batch
        r_x0 = self.r_x0
        s_x0 = self.s_x0

        if self.transform == 'softplus':
            fval_org = fval.copy()
            fval = np.log1p(np.exp(fval_org))
        elif self.transform == 'none':
            fval = fval + 1e-50

        if x_batch is not None:
            log_fval = np.log(fval)
            h_vals = self._hammer_function(x, x_batch, r_x0, s_x0)
            log_fval += h_vals.sum(axis=-1)
            fval = np.exp(log_fval)
        return fval
    
    
    
    
    
    def _cone_function_precompute(self):
        x0 = self.x_batch
        L = self.L
        M = self.best
        mu, var = self.surrogate.predict(x0)
        r_mu = (mu.flatten() - M) / L
        r_std = np.sqrt(var.flatten()) / L

        r_mu = r_mu.flatten()
        r_std = r_std.flatten()
        return r_mu, r_std

    def _cone_function(self, x, x0):
        """
        Creates the function to define the exclusion zones
        Using half the Lipschitz constant as the gradient of the penalizer.
        We use the log of the penalizer so that we can sum instead of multiply
        at a later stage.
        """
        r_mu = self.r_mu
        r_std = self.r_std

        x_norm = np.sqrt(np.square(np.atleast_2d(x)[:, None, :] - np.atleast_2d(x0)[None, :, :]).sum(-1))
        norm_jitter = 0  # 1e-100
        # return 1 / (r_mu + r_std).reshape(-1, len(x0)) * (x_norm + norm_jitter)
        return 1 / (r_mu + r_std) * (x_norm + norm_jitter)
    
    
    def _penalized_acquisition(self, x):
        '''
        Creates a penalized acquisition function using the 4th norm between
        the acquisition function and the cone
        '''
        fval = self.acq.evaluate(x)
        x_batch = self.x_batch

        if self.transform == 'softplus':
            fval_org = fval.copy()
            fval = np.log1p(np.exp(fval_org))
        elif self.transform == 'none':
            fval = fval + 1e-50

        if x_batch is not None:
            h_vals = self._cone_function(x, x_batch).prod(-1)
            h_vals = h_vals.reshape([1, -1])
            clipped_h_vals = np.linalg.norm(np.concatenate((h_vals, np.ones(h_vals.shape)), axis=0), -5, axis=0)

            fval *= clipped_h_vals

        return fval
    
    
    
    
    
    