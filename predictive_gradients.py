#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:04:10 2020

@author: andrew
"""

import numpy as np
import GPy

import matplotlib.pyplot as plt

x_test = np.linspace(0.3, 1.4,100).reshape(-1,1)
mf = GPy.mappings.Linear(1,1)
X = np.linspace(0.3, 1.4, 100).reshape(-1,1)
Y = (6*X - 2)**2 * np.sin(12*X - 4)
#gp_model = GPy.models.GPRegression(X, Y, mean_function=mf)

gp_model = GPy.models.GPRegression(X, Y)
gp_model.optimize_restarts()
#print(gp_model)
#print(gp_model.predict(x_test)[0])
print(max(gp_model.predictive_gradients(x_test)[0]))

plt.plot(x_test, gp_model.predict(x_test)[0])
#plt.plot(x_test, gp_model.predictive_gradients(x_test)[:][0])