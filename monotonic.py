#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:57:42 2019

@author: andrew
"""

from scipy.stats import beta as B
from scipy.stats import dirichlet as D
import numpy as np
EXT_BETA_CDF = lambda x, a, b : np.array([B.cdf(x, a[i],b[i]) for i in range(len(a))])

class MonotonicBetaCDF(object):

    def __init__(self, n_betas, alphas=None, betas=None, omegas=None, \
                            alb=0, aub=5, blb=0, bub=5, wub=1, wlb=0):
        self.nb = n_betas
        self.alb = alb
        self.aub = aub
        self.blb = blb
        self.bub = bub
        self.wub = wub
        self.wlb = wlb
        if alphas is not None:
            self.alphas = alphas
            self.betas = betas
            self.omegas = omegas
        else:
            self.alphas, self.betas, self.omegas = self.random_sample()

    def set_alphas(self, alphas):
        self.alphas = alphas

    def set_betas(self, betas):
        self.betas = betas

    def set_omegas(self, omegas):
        self.org_omegas = omegas
        self.omegas = omegas/np.sum(omegas)

    def random_sample(self):
        # limits of alpha: alb, aub
        # limts of beta: blb, bub
        # number of beta values: nb
        alphas = np.random.random_sample(self.nb) * (self.aub - self.alb) + self.alb
        print("alphas", alphas)
        
        betas = np.random.random_sample(self.nb) * (self.bub - self.blb) + self.blb
        alphas = np.ones(len(betas))
        # Random sample with Dirichlet distribution
        omegas = D.rvs([1]*self.nb)[0]
        return alphas, betas, omegas

    def evaluate(self, n_samples, positions, lb, ub):
        #positions = (np.arange(0,n_samples) + 1)/(n_samples + 1)
        #positions = (np.arange(n_samples))/(n_samples-1.0)
        print(positions)
        print(np.dot(self.omegas, EXT_BETA_CDF(positions, self.alphas, self.betas)))
        return np.dot(self.omegas, EXT_BETA_CDF(positions, self.alphas, self.betas))*(ub-lb) + lb
    
    
#vert_origin = 0 #doesn't seem to influence
#n_rows = 3
#vert_positions = np.array([-0.2, 0, 0.2])
#xlb, xub = -0.2, 3.25*0.2
#rlb, rub = 0.005, 0.5*0.2
#nlb, nub = 1, 5
#n_coeffs_radii = [3]*n_rows
#n_coeffs_num = 4
        
# 1 extra parameter for beta function
n_betas = 3
angleub = 45.0


centers = MonotonicBetaCDF(n_betas)    

# this is constant:
n_samples = 5

# this is constant:   
anglelb = 0.0


y = np.array([-14.9375, -10.875, -6.0, -2.34375, 1.3125])   # top of tray

# this is constant:
#positions = (np.arange(n_samples))/(n_samples-1.0)

positions = (y-1.3125)/(-14.9375-1.3125)

angle = centers.evaluate(n_samples, positions, anglelb, angleub)

import matplotlib.pyplot as plt
plt.ion()
#plt.ioff()

plt.figure()
plt.plot(y,angle, marker="o")
plt.plot(0.8375, angle[3],marker="x", color='red' )
plt.draw()


