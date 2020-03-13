#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:03:51 2020

@author: andrew
"""
from scipy.stats import norm as norm2
import numpy as np
import matplotlib.pyplot as plt

def _hammer_function(x, x0, r_x0, s_x0):
    '''
    Creates the function to define the exclusion zones
    '''
    return norm2.logcdf((np.sqrt((np.square(np.atleast_2d(x)[:,None,:]-np.atleast_2d(x0)[None,:,:])).sum(-1))- r_x0)/s_x0)




L = 311.55067041
r_x0 = (0.00454236*(20*L)/(L))/50

s_x0 = 5.12674034/(L)

x0 = 1.4

tx = np.linspace(0.3, 1.4, 20000)[:,None]

fig, ax1 = plt.subplots(1,1, figsize=(8,8))
fig.subplots_adjust(hspace=0.25)

ax1.scatter(tx, _hammer_function(tx, x0, r_x0, s_x0), c="red", alpha=0.75)

