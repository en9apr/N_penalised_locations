#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:35:17 2020

@author: andrew
"""

import GPy
GPy.tests()

import numpy as np

X = np.random.uniform(-5, 5, (20,1))
Y = np.array([0.25 * (a*a) + (.25 * np.random.randn()) for a in X])

kernel = GPy.kern.RBF(input_dim=1)
m = GPy.models.GPRegression(X,Y,kernel)
print(m)
m.plot()