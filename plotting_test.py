# coding: utf-8
#imports
import matplotlib.pyplot as plt
plt.ion()
import numpy as np
from surrogate_test import Surrogate
import GPy as gp
# objective sense
sense = -1 # 1 for minimisation 
# get data
fn = 'hydro_1D_HypI_b21s11_r20190822.npz'#"hydro_1D_EGO_b21s11_r180817.npz"#"hydro_1D_EGO_b20s10_r0.npz"
data = np.load(fn)
x = data['arr_0']
x_org = x.copy()
y = data['arr_1']#[:,None]
#y = y[:, None]
x = x[:y.shape[0]] # just get the ones that have been evaluated; may not be necessary if x.shape = y.shape
# define kernel
kern = gp.kern.Matern52(input_dim=1)
# build model
surr = Surrogate(x, y, kern)
# plot
plt.figure(figsize=(6,10))
plt.subplot(211)
plt.scatter(x, sense*y,marker="x", color='red' )
plt.xlim(10, 17)
xtest = np.linspace(10, 17, 2000)[:,None]
xtest
xbase = 13.125
ybase = 0.625348
ybase_true = 0.55
yp, sp = surr.predict(xtest)
ei = surr.expected_improvement(xtest, obj_sense=-1, lb=np.ones(1)*10, ub=np.ones(1)*17)
ind = np.argmax(ei)
plt.plot(xtest, sense*yp, color="green")
plt.axvline(x=xtest[ind], color="red", ls="dashed", lw=2, alpha=0.5)
plt.fill_between(np.squeeze(xtest), np.squeeze(sense*(yp+sp)), np.squeeze(sense*(yp-sp)), color="green", alpha=0.25)
plt.axvline(x=xbase, ls="dashed", lw=2, alpha=0.5, color="gray")
plt.axhline(y=ybase, ls="dashed", lw=2, alpha=0.5, color="gray")
plt.axhline(y=ybase_true, ls="dashed", lw=2, alpha=0.5, color="blue")
plt.xlabel('Radius (inches)')
plt.ylabel('Collection efficiency (%)')
plt.subplot(212)
plt.plot(xtest, ei, color="blue")
plt.xlim(10, 17)
plt.xlabel('Radius (inches)')
plt.ylabel('Expected improvement')
plt.axvline(x=xtest[ind], color="red", ls="dashed", lw=2, alpha=0.5)
plt.tight_layout()
plt.savefig("surrogate.png", bbox_inches='tight')

