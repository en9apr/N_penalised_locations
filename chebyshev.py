#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:57:42 2019

@author: andrew
"""

from numpy.polynomial.chebyshev import Chebyshev as C
import numpy as np

class Chebyshev(object):

    def __init__(self, n_coeffs, coeffs=None, clb=-1, cub=1, domain=[0,1], window=[-1,1]):
        self.nc = n_coeffs
        self.domain = domain
        self.window = window
        self.clb = clb
        self.cub = cub
        # determine ymin
        cf = np.array([self.clb]*n_coeffs)
        self.coeffs = cf
        self.update_function()
        self.ymin = self.function(1)
        # determine ymax
        cf = np.array([self.cub]*n_coeffs)
        self.coeffs = cf
        self.update_function()
        self.ymax = self.function(1)
        # set coefficients
        if coeffs is None:
            self.coeffs = self.random_sample()
        else:
            self.coeffs = coeffs
        self.update_function()

    def update_function(self):
        assert self.nc == len(self.coeffs)
        self.function = C(self.coeffs, domain=self.domain, window=self.window)

    def random_sample(self):
        return np.random.random_sample(self.nc) * (self.cub - self.clb) + self.clb

    def evaluate(self, n_samples, positions, lb, ub):
        # find positions
        #positions = (np.arange(n_samples) + 1)/(n_samples + 1)
        # compute function values
        y = self.function(positions)
        # scale between 0 and 1
        y = (y - self.ymin)/(self.ymax - self.ymin)
        # scale between lb and ub
        y = ((ub - lb) * y) + lb
        return y
    
#vert_origin = 0 #doesn't seem to influence
#n_rows = 3
#vert_positions = np.array([-0.2, 0, 0.2])
#xlb, xub = -0.2, 3.25*0.2
#rlb, rub = 0.005, 0.5*0.2
#nlb, nub = 1, 5
#n_coeffs_radii = [3]*n_rows
#n_coeffs_num = 4
        
# 1 extra parameter for beta function
#n_betas = 

n_coeffs_num = 3

centers = Chebyshev(n_coeffs_num)    

# this is constant:
n_samples = 5

# this is constant: 
  
zlb = -4.0
zub = 4.0

y = np.array([-14.9375, -10.875, -6.0, -2.34375, 1.3125])   # top of tray

# this is constant:
#positions = (np.arange(n_samples))/(n_samples-1.0)

positions = (y-1.3125)/(-14.9375-1.3125)

centre_location = centers.evaluate(n_samples, positions, zlb, zub)

import matplotlib.pyplot as plt
plt.ion()
#plt.ioff()

plt.figure()
plt.plot(y, centre_location, marker="o")
plt.plot(0.8375, centre_location[3],marker="x", color='red' )
plt.draw()


