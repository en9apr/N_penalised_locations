#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:54:30 2020

@author: andrew
"""
from scipy.special import erf as ERF
import numpy as np
import matplotlib.pyplot as plt


def transform_sequence(n, lb=0, ub=185, scale=0.01, lw=0.25, uw=0.75):
    np = (n/ub * (ub -lb)) - ((ub -lb)/2)
    nub = (ub -lb)/2
    nlb = -(ub-lb)/2
    w = (ERF(scale*np) - ERF(scale*nlb))/(ERF(scale*nub) - ERF(scale*nlb))
    w = (w * (uw - lw)) + lw
    return w


weight=np.zeros([2000])
n=np.zeros([2000])


for i in range(2000):
    weight[i]=transform_sequence(i)
    n[i] = i
    
plt.figure(1)
plt.cla()
plt.scatter(n, weight, marker="x", color="blue", alpha=0.35)
'''
circle = plt.Circle((0,0), 1, color="blue", alpha=0.25)
ax = plt.axes()
ax.add_artist(circle)
plt.axvline(x=0, color="black", alpha=0.15)
plt.axhline(y=0, color="black", alpha=0.15)        
'''
plt.draw()
plt.pause(0.005)    