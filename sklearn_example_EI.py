# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Author: Vincent Dubourg <vincent.dubourg@gmail.com>
#         Jake Vanderplas <vanderplas@astro.washington.edu>
#         Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>s
# License: BSD 3 clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from numpy.linalg import norm

from scipy.special import erf as ERF

np.random.seed(1)


def f(x):
    """The function to predict."""
    return x * np.sin(x)

def max_L(y, x):
    grad_y_x = np.gradient(y[:,0], x[:,0])
    res = max(np.sqrt(grad_y_x*grad_y_x))
    if res<1e-7: res=10
    return res 


def downweight(L, radius, s, l2norm, p, gamma):
    '''
    Quantiles of the Gaussian distribution useful to determine the acquisition function values
    :param acquisition_par: parameter of the acquisition function
    :param fmin: current minimum.
    :param m: vector of means.
    :param s: vector of standard deviations.
    '''

    if isinstance(s, np.ndarray):
        s[s<1e-10] = 1e-10
    elif s< 1e-10:
        s = 1e-10
        
    Phi =  (( (l2norm+1e-12) / (radius + (gamma * (s/L)) ))**p + (1.0**p) )**(1.0/p)
    return Phi 










# ----------------------------------------------------------------------
#  First the noiseless case
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T

# Observations
y = f(X).ravel()

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)




#lb = 0.0
#ub = 10.0
##tx = np.linspace(lb, ub, 500)[:,None]
##pred_y, pred_s = mop.surr.predict(tx)
#p = -5
#gamma = 1.0
#
#
#L = max_L(y_pred, x)
#Min = min(y_pred)
#
#x_new=4
#
#yinds, std_dev_inds = gp.predict(x_new, return_std=True)
#radius = abs(Min - yinds) /L + gamma*(std_dev_inds / L)
#l2norm = norm(x_new-x, axis=1)
#phi = downweight(L, radius, std_dev_inds, l2norm, p, gamma)
#
#print("Plotting: min phi", min(phi[0]))

obj_sense = -1

f_best = obj_sense * np.max(obj_sense * y_pred)
u =  obj_sense * (y_pred - f_best) / sigma

# normal cumulative distribution function
PHI = (0.5 * ERF(u/np.sqrt(2.0)))+0.5 
# normal density function
phi = 1/np.sqrt(2.0*np.pi)*np.exp(-u**2/2.0)   

ei =  sigma * ((u * PHI) + (phi))
















# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')


plt.figure()
plt.plot(x, ei, 'r', label=r'$EI(x)$')
plt.xlabel('$x$')
plt.ylabel('$EI(x)$')
plt.legend(loc='upper right')




