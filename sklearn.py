#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 20:45:53 2020

@author: andrew
"""

import sklearn.gaussian_process as gp

import numpy as np

# X_tr <-- training observations [# points, # features]
# y_tr <-- training labels [# points]
# X_te <-- test observations [# points, # features]
# y_te <-- test labels [# points]

kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))

model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)


X_tr = np.random.uniform(-5, 5, (20,1))
Y_tr = np.array([0.25 * (a*a) + (.25 * np.random.randn()) for a in X_tr])

model.fit(X_tr, Y_tr)
params = model.kernel_.get_params()


X_te = np.random.uniform(-5, 5, (20,1))
Y_te = np.array([0.25 * (a*a) + (.25 * np.random.randn()) for a in X_te])


y_pred, std = model.predict(X_te, return_std=True)

MSE = ((y_pred-Y_te)**2).mean()
