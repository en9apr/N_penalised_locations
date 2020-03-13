import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure(1)
plt.cla()

#n_samples = 31
data = np.load('initial_samples.npz')
A = data['arr_0']
B = data['arr_1']
n_samples = len(B)

b = B.copy() #mop.m_obj_eval(xtr)
plt.scatter(b[:n_samples,0], b[:n_samples,1], marker="x", color="red", alpha=0.35)





#
#data2 = np.load('evaluate_HypI_b409s109_r0.npz')
#X = data2['arr_0']
#Y = data2['arr_1']
##n_samples = len(Y)
#
#y = Y.copy() #mop.m_obj_eval(xtr)
##plt.scatter(y[:n_samples,0], y[:n_samples,1], marker="x", color="blue", alpha=0.35)
#
#
#
#
#
#cs = plt.scatter(y[n_samples:,0], y[n_samples:,1], c=np.arange(1, y.shape[0]-n_samples+1, 1), alpha=0.35)

#cbar = fig.colorbar(cs)
#plt.ylim(-1, -0.8)
#plt.xlim(0, 1)



#plt.scatter(y[-1,0], y[-1,1], facecolor="none", edgecolor="black", s=80)
plt.xlabel("$f_1$ - Total Collection Efficiency")
plt.ylabel("$f_2$ - Static Overflow Pressure Drop")
plt.draw()
plt.pause(0.005)
plt.savefig('Initial_Sampling.png')
