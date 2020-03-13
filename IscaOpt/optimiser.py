#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Optimiser suite
================================================================================
:Author:
   Alma Rahat   <A.A.M.Rahat@exeter.ac.uk>
:Date:
   19 April 2017
:Copyright:
   Copyright (c)  Alma Rahat, University of Exeter, 2017
:File:
   optimiser.py
"""

# imports
import numpy as np
import GPy as GP
import cma as CMA
try:
    import mono_surrogate
    import multi_surrogate
except:
    import IscaOpt.mono_surrogate as mono_surrogate
    import IscaOpt.multi_surrogate as multi_surrogate
import sys, time
import matplotlib.pyplot as plt

# APR added:
import subprocess
try: 
    from surrogate import Surrogate
except:
    from .surrogate import Surrogate
    
from numpy.linalg import norm
from scipy.special import erf as ERF    

def get_quantiles(L, radius, s, l2norm, p, gamma):
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
    
    Phi =  ( ( (l2norm+1e-12) / (radius + (gamma * (s/L)) ))**p + (1.0**p) )**(1.0/p)
    return Phi


def mod_evaluate(x, toolbox):
    """
    Modify toolbox function to work with CMA_ES (Hansen). In CMA-ES the default 
    is to minimise a given function. Here we negate the given function to turn 
    it into a maximisation problem.
    
    Parameters.
    -----------
    x (np.array): decision vector
    toolbox (DEAP toolbox): toolbox with the infill cost function.
    """
    try:
        return -toolbox.evaluate(x)[0][0]
    except:
        return -toolbox.evaluate(x)[0]

class Optimiser(object):
    """
    A suite of optimisers where the inidividual optimisers may be called
    without creating an instance of the optimiser.
    """
        
    @staticmethod
    def CMA_ES(toolbox, centroid=[1], std_dev=1, cma_options={}):
        """
        A wrapper for CMA-ES (Hansen) with DEAP toolbox.
        
        Parameters.
        -----------
        toolbox (DEAP toolbox): an appropriate toolbox that consists of 
                                the objective function for CMA-ES.
        centroid (list or str): centroid vector or a string that can produce a 
                                centroid vector. 
        std_dev (float): standard deviation from the centroid.
        lb (list or np.array): lower bounds in the decision space. 
        ub (list or np.array): upper bounds in the decision space.
        cma_options (dict): cma_options from Hansen's CMA-ES. Consult the relevant 
                            documentation.
                            
        Returns the best approximation of the optimal decision vector.
        """
        func = mod_evaluate
        fargs = (toolbox,)
        res = CMA.fmin(func, centroid, std_dev, \
            options=cma_options, args=fargs, bipop=True, restarts=9)
        return res[0]
        
    @staticmethod
    def grid_search_1D(toolbox, lb, ub, n=10000, obj_sense=1):
        '''
        Grid search in one dimension. Evluate the objective function at equally
        spaced locations between the lower and upper boundary of the decision
        space.
        
        Parameters.
        -----------
        toolbox (DEAP toolbox): toolbox that contains the objective function.
        lb (np.array): lower bounds on the decision space.
        ub (np.array): upper bounds on the decision space.
        n (int): number of samples to evaluate.
        obj_sense (int): optimisation sense. Keys. 
                            -1: minimisation.
                             1: maximisation.
                             
        Returns the best decision vector from the observed solutions. 
        '''
        x = np.linspace(lb, ub, n)[:,None]
        #y = np.array([toolbox.evaluate(i) for i in  x])
        y = toolbox.evaluate(x)
        opt_ind = np.argmax(obj_sense*y)
        return x[opt_ind]
    
    @staticmethod
    def EMO(func, fargs=(), fkwargs={}, cfunc=None, cargs=(), ckwargs={},\
            settings={}):
        '''
        Optimising a single- or multi-objective problem using 
        Gaussian process surrogate(s).
        
        Parameters. 
        -----------
        func (function): objective function.
        fargs (tuple): arguments to the objective function.
        fkwargs (dict): keyword arguments for the objective function.
        cfunc (function): cheap constraint function.
        cargs (tuple): arguments to the constraint function.
        ckwargs (dict): keyword arguments to the constraint fucntion.
        settings (dict): various settings for the Bayesian optimiser. 
        
        Returns a list of all the observed decision vectors, the associated 
        objective values, and relevant hypervolume.
        '''
        print("EMO(): start of function")
        
        start_sim = time.time()
        # parameters for problem class
        n_dim = settings.get('n_dim', 2)
        n_obj = settings.get('n_obj', 2)
        lb = settings.get('lb', np.zeros(n_dim))
        ub = settings.get('ub', np.ones(n_dim))
        ref_vector = settings.get('ref_vector', [150.0]*n_obj)
        obj_sense = settings.get('obj_sense', [-1]*n_obj) # default: all minimisation; 
                                                          # haven't tried maximisation, but should work.
        method_name = settings.get('method_name', 'HypI')
        visualise = settings.get('visualise', False)
        # parameters for EGO
        n_samples = settings.get('n_samples', n_dim * 10) # default is 10 
                                                          # initial samples per dimension
        budget = settings.get('budget', n_samples+5)
        kern_name = settings.get('kern', 'Matern52')
        verbose = settings.get('verbose', True)
        svector = settings.get('svector', 15)
        maxfevals = settings.get('maxfevals', 20000*n_dim)
        multisurrogate = settings.get('multisurrogate', False)
        # history recording
        sim_dir = settings.get('sim_dir', '')
        run = settings.get('run', 0)
        draw_true_1d = settings.get('draw_true_1d', False)
        # cma_options for Hansen's CMA-ES
        cma_options = settings.get('cma_options', \
                                    {'bounds':[list(lb), list(ub)], \
                                     'tolfun':1e-7, \
                                     'maxfevals':maxfevals,\
                                     'verb_log': 0,\
                                     'CMA_stds': np.abs(ub - lb)}) 
        cma_centroid = '(np.random.random('+str(n_dim)+') * np.array('+\
                        str(list(ub - lb))+') )+ np.array('+str(list(lb))+')'
        cma_sigma = settings.get('cma_sigma', 0.25)
        
        # initial design file
        init_file = settings.get('init_file', None)
        X = None
        Y = None
        
        # intial training data
        if init_file is not None:
            data = np.load(init_file)
            X = data['arr_0']
            Y = data['arr_1']
            xtr = X.copy()
            print('Training data loaded from: ', init_file)
            print('Training data shape: ', xtr.shape)
        if verbose:
            print ("=======================")
            
        # initialise
        if verbose:
            print('Simulation settings. ')
            print(settings)
            print('=======================')
        hpv = []
        
        print("EMO(): setting of initial values completed")
        
        # determine method
        if multisurrogate:
            method = getattr(multi_surrogate, method_name)
            kern = [getattr(GP.kern, kern_name)(input_dim=n_dim, ARD=True) \
                        for i in range(n_obj)]
        else:
            method = getattr(mono_surrogate, method_name)
            kern = getattr(GP.kern, kern_name)(input_dim=n_dim, ARD=True)
        print(method, kern)
        
        print("EMO(): method selected " + str(method))
        
        # method specific kwargs
        skwargs = {}
        if method_name == 'HypI':
            skwargs['ref_vector'] = ref_vector
        if method_name == 'ParEGO':
            skwargs['s'] = svector
        if method_name == 'MPoI' or method_name == 'SMSEGO':
            skwargs['budget'] = budget
        print('method used: ', method.__name__)
        
        print("EMO(): method used" + method_name)
        # EGO
        i = 1
        count_limit = (budget-n_samples) # budget is 21, n_samples is 11, so count_limit is 10
        sim_file = sim_dir + func.__name__ + '_' + method_name + \
                                        '_b' + str(budget) + \
                                        's' + str(n_samples) \
                                        + '_r' + str(run) + '.npz'
                                        
        print("EMO(): sim_file name defined")
        
        # Phi failures is initially one
        phi_failure = np.ones(20000)[:,None]
        tx = np.linspace(lb, ub, maxfevals)[:,None]
        
        r_small = 1.0e-3
        
        failure_gradient_weighting = 20.0
        parallel_gradient_weighting = 0.25
        
        while True:
            
            global_failed_indices = []
            print('============================================================')
            print('EPISODE: ', i)
            print('============================================================')

            # to count samples
            jj = 0

            mop = method(func,n_dim, n_obj, lb, ub, obj_sense=obj_sense, \
                        args=fargs, X=X, Y=Y, kwargs=fkwargs, kern=kern,\
                        ref_vector=ref_vector)

            # Print the bounds
            print("EMO(): bounds" + str(mop.lower_bounds) + str(mop.upper_bounds) )
            
            if i == 1 and init_file is None:
                print("EMO(): number of samples for Latin Hypercube", str(n_samples))
                xtr = mop.lhs_samples(n_samples, cfunc, cargs, ckwargs)
                print("EMO(): Latin Hypercube samples completed")
                if xtr.shape[0] < n_samples:
                    count_limit += n_samples - xtr.shape[0] - 1
                    print("adjusted count limit: ", count_limit)
                    print("initial samples: ", xtr.shape[0])
                    print("EMO(): adjusted count limit: " + str(count_limit))
                    print("EMO(): samples from Latin Hypercube: " + str(xtr.shape[0]))
            
            print("EMO(): before toolbox")   

            # Phi parallel is set to one everytime a toolbox is created
            phi_parallel = np.ones(20000)[:,None]        
            
            # Surrogate runs hammer function precompute
            toolbox = mop.get_toolbox(xtr, phi_failure, phi_parallel, global_failed_indices, skwargs, cfunc=cfunc, cargs=cargs, ckwargs=ckwargs)
            
            print('===================== Failed Locations =====================')
            print("EMO(): global_failed_indices: ", mop.surr.global_failed_indices)
            print("EMO(): success", mop.surr.success)
            
            if(mop.surr.success == False):
                
                # Penalise all failures
                for kkk in range(len(mop.surr.global_failed_indices[0])):
                    fail_location = np.array([mop.surr.global_failed_indices[0][kkk]])
                    radius, std_dev_inds = mop.surr._hammer_function_precompute(fail_location, failure_gradient_weighting*mop.surr.L, mop.surr.best, mop.surr)
                    
                    hammer = mop.surr._hammer_function(tx, fail_location, radius, std_dev_inds)
                    hammer = hammer[:, None]
                    phi_failure = phi_failure*hammer 
                    
                    print("EMO(): Error handling: At the current value: ", kkk, " x = ", mop.surr.global_failed_indices[0][kkk], " min(phi_failure) ", min(phi_failure))            
                    print("EMO(): Error handling: global_failed_indices_2[kkk]: ", mop.surr.global_failed_indices[0][kkk])
                    print("EMO(): Error handling: radius: ", radius)
                    print("EMO(): Error handling: std_dev_inds: ", std_dev_inds*failure_gradient_weighting*mop.surr.L)
                    print("EMO(): Error handling: mop.surr.L: ", mop.surr.L)
                    print("EMO(): Error handling: mop.surr.best: ", mop.surr.best)
                    print("EMO(): Error handling: min(phi_failure)", min(phi_failure))
            print('============================================================')
            
            # Assignment to mop.surr.phi_failure is required for penalized_acquisition function to work
            mop.surr.phi_failure = phi_failure
            
            # EI computed on the basis of mop.surr.phi_failure only
            ei_parallel_1 =  mop.surr.penalized_acquisition(tx, obj_sense=-1, lb=lb, ub=ub, cfunc=cfunc, cargs=cargs, ckwargs=ckwargs)
            
            print("EMO(): after toolbox")   
            
            hpv.append(mop.current_hv)
            print("EMO(): after hpv append")  
            
            X = mop.X.copy()
            Y = mop.Y.copy()
            print("EMO(): X and Y values defined")
            
            if i > count_limit:
                break
            i += 1
            
            ###############
            # First Sample:
            ###############

            # Search for new optimimum location
            if n_dim > 1:
                xopt1 = Optimiser.CMA_ES(toolbox, cma_centroid, cma_sigma, cma_options=cma_options)
            else:
                xopt1 = Optimiser.grid_search_1D(toolbox, lb, ub, n=maxfevals)
            
            # new sample point location from CMA-ES optimisation
            x_new = np.reshape(xopt1, (1, -1))
            
            # make a copy of the old training data
            xtr = mop.X.copy()
            
            # include new sample in the training data.
            xtr = np.concatenate([xtr, x_new])
            print("EMO(): Next point chosen")

            y = Y.copy()
            xtmp = X.copy()
             
            print('===================== Surrogate values =====================')
            
            # Print out values fixed for this episode
            print("Plotting: L: ", mop.surr.L)
            print("Plotting: Min: ", mop.surr.best)
            
            # Values that are constant for this episode:
            pred_y, pred_s = mop.surr.predict(tx)
            
            print('===================== xopt1: plotting ======================')
                      
            # New values according to x_new prediction
            r_x0, s_x0 = mop.surr._hammer_function_precompute(x_new, np.amin(parallel_gradient_weighting*mop.surr.L), mop.surr.best, mop.surr)
            mop.surr.r_x0 = np.array(r_x0)
            mop.surr.s_x0 = np.array(s_x0)
            
            
#            y_new, std_new = mop.surr.predict(x_new)
#            r_new = (abs(mop.surr.best - y_new) / (parallel_gradient_weighting*mop.surr.L)) 
            
            # Print out old and new values
            jj += 1
            print("Plotting: Sample One = ", jj)
            print("Plotting: x_old: ", mop.surr.xinds)
            print("Plotting: r_old: ", mop.surr.radius)
            print("Plotting: std_old: ", mop.surr.std_dev_inds)
            print("Plotting: x_new: ", x_new)
            print("Plotting: r_new: ", r_x0)
            print("Plotting: std_new: ", s_x0)
            print('============================================================')
                    
            # Exact solutions
            exact_solution = (6*tx - 2)**2 * np.sin(12*tx - 4)
            
            ###################
            # Creation of plots
            ###################
            
            # Create figures
            fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))
            fig.subplots_adjust(hspace=0.25)
            y = Y.copy()
            xtmp = X.copy()
            
            # Surrogate model
            ax1.scatter(xtmp[:n_samples], y[:n_samples], marker="x", color="red", alpha=0.75)
            ax1.scatter(xtmp[n_samples:], y[n_samples:], c="red", alpha=0.75)
            ax1.axvline(x=x_new, color="red", ls="dashed", lw=2, alpha=0.5)
            ax1.axhline(y=mop.surr.best, color="green", ls="dashed", lw=2, alpha=0.5)
            ax1.plot(tx, exact_solution, color="black")
            ax1.plot(tx, pred_y, color="green")
            ax1.fill_between(np.squeeze(tx), np.squeeze((pred_y)-pred_s), np.squeeze((pred_y)+pred_s), color="green", alpha=0.3)
            ax1.fill_betweenx(np.array([-40, 30]), x_new[0]-r_x0[0], x_new[0]+r_x0[0], color="red", alpha=0.1)          
            ax1.set_xlabel('x')
            ax1.set_ylabel('f(x)')
            ax1.set_ylim(-40, 30)
            ax1.set_xlim(0.2, 1.5)
            
            # Expected improvement
            ax2.plot(tx, ei_parallel_1, color="green")
            ax2.axvline(x=x_new, color="red", ls="dashed", lw=2, alpha=0.5)
            ax2.set_xlabel('x')
            ax2.set_ylabel('EI(x)')
            ax2.set_xlim(0.2, 1.5)
            
            # Penalisation on second axis
            ax3 = ax2.twinx()
            ax3.plot(tx, phi_failure, color="red")
            ax3.plot(tx, phi_parallel, color="blue")           
            ax3.fill_betweenx(np.array([-40, 30]), x_new[0]-r_x0[0], x_new[0]+r_x0[0], color="red", alpha=0.1)
            ax3.set_ylim(-0.1, 1.1)
            ax3.set_ylabel('phi(x)')
            ax3.set_ylim(-0.05, 1.05)

            # Save as png
            plt.pause(0.005)
            plt.show()
            plt.tight_layout()
            fig.savefig('Episode' + '_' + str(i-1) + '_' 'xopt1'+ '.png', transparent=False)
                    
            if(r_x0 < r_small):
                print('===================== xopt2: plotting  =====================')
                print("Plotting: radius is less than 1mm, no more samples are taken")  
                print('============================================================')
                
            else:    
                
                ################
                # Second Sample:
                ################
    
                # New is old
                mop.surr.xinds = x_new
                mop.surr.radius = r_x0
                mop.surr.std_dev_inds = s_x0
                
                # New penalisation
                hammer = mop.surr._hammer_function(tx, x_new, r_x0, s_x0)
                hammer = hammer[:, None]
                phi_parallel = phi_parallel*hammer
                mop.surr.phi_parallel = phi_parallel
                
                # store them
                if mop.surr.pen_locations is None:
                    mop.surr.pen_locations = np.atleast_2d(x_new)
                    mop.surr.r_x0 = np.array(r_x0)
                    mop.surr.s_x0 = np.array(s_x0)
                    
                else:
                    mop.surr.pen_locations = np.concatenate((mop.surr.pen_locations, np.atleast_2d(x_new)))
                    mop.surr.r_x0 = np.concatenate((mop.surr.r_x0, r_x0))
                    mop.surr.s_x0 = np.concatenate((mop.surr.s_x0, s_x0))
                    #print("mop.surr.pen_locations", mop.surr.pen_locations)

                # calculate hammer function for failures
                if not mop.surr.success:
                    N = len(mop.surr.global_failed_indices[0]) ## index needed for 1D
                    print("N", N)
                    r_x0 = np.zeros(N)
                    s_x0 = np.zeros(N)
                    x0 = np.zeros((N, n_dim))
                    
                    for kkk in range(N):
                        x0[kkk] = np.array([mop.surr.global_failed_indices[0][kkk]])
                        r_x0[kkk], s_x0[kkk] = mop.surr._hammer_function_precompute(x0[kkk], failure_gradient_weighting*mop.surr.L, mop.surr.best, mop.surr)
    
                    mop.surr.pen_locations = np.concatenate((mop.surr.pen_locations, x0))
                    mop.surr.r_x0 = np.concatenate((mop.surr.r_x0, r_x0))
                    mop.surr.s_x0 = np.concatenate((mop.surr.s_x0, s_x0))                
                

#                for kkk in range(len(mop.surr.global_failed_indices[0])):
#                    fail_location = np.array([mop.surr.global_failed_indices[0][kkk]])
#                    radius, std_dev_inds = mop.surr._hammer_function_precompute(fail_location, failure_gradient_weighting*mop.surr.L, mop.surr.best, mop.surr)
#                    
#                    hammer = mop.surr._hammer_function(tx, fail_location, radius, std_dev_inds)
#                    hammer = hammer[:, None]
#                    phi_failure = phi_failure*hammer 



                
                
                
#                radius, std_dev_inds = mop.surr._hammer_function_precompute(x_new, parallel_gradient_weighting*mop.surr.L, mop.surr.best, mop.surr)
#                hammer = mop.surr._hammer_function(tx, x_new, radius, std_dev_inds)
#                hammer = hammer[:, None]
#                phi_parallel = phi_parallel*hammer
#                mop.surr.phi_parallel = phi_parallel
                
                # Penalise EI
                ei_parallel_2 = mop.surr.penalized_acquisition(tx, obj_sense=-1, lb=lb, ub=ub, cfunc=cfunc, cargs=cargs, ckwargs=ckwargs)
                
                # Search for new optimimum location                     
                if n_dim > 1:
                    xopt2 = Optimiser.CMA_ES(toolbox, cma_centroid, cma_sigma, \
                                        cma_options=cma_options)
                else:
                    xopt2 = Optimiser.grid_search_1D(toolbox, lb, ub, n=maxfevals)
                
                # New sample point location from CMA-ES optimisation
                x_new = np.reshape(xopt2, (1, -1))

                print('===================== xopt2: plotting  =====================')
                
                # New values according to x_new prediction
                
                r_x0, s_x0 = mop.surr._hammer_function_precompute(x_new, np.amin(parallel_gradient_weighting*mop.surr.L), mop.surr.best, mop.surr)
#                y_new, std_new = mop.surr.predict(x_new)
#                r_new = (abs(mop.surr.best - y_new) / (parallel_gradient_weighting*mop.surr.L)) 
                
                # Print out new and old values
                jj += 1
                print("Plotting: Sample Two = ", jj)
                print("Plotting: x_old: ", mop.surr.xinds)
                print("Plotting: r_old: ", mop.surr.radius)
                print("Plotting: std_old: ", mop.surr.std_dev_inds)
                print("Plotting: x_new: ", x_new)
                print("Plotting: r_new: ", r_x0)
                print("Plotting: std_new: ", s_x0)
                print("Plotting: min(phi_parallel)", min(phi_parallel))
                print('============================================================')
                
                # include new sample in the training data.
                xtr = np.concatenate([xtr, x_new])
            
                ###################
                # Creation of plots
                ###################
                
                # Create figures
                fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))
                fig.subplots_adjust(hspace=0.25)
                y = Y.copy()
                xtmp = X.copy()
                
                # Surrogate model
                ax1.scatter(xtmp[:n_samples], y[:n_samples], marker="x", color="red", alpha=0.75)
                ax1.scatter(xtmp[n_samples:], y[n_samples:], c="red", alpha=0.75)
                ax1.axvline(x=xopt1, color="gray", ls="dashed", lw=2, alpha=0.5)
                ax1.axvline(x=xopt2, color="red", ls="dashed", lw=2, alpha=0.5)
                ax1.axhline(y=mop.surr.best, color="green", ls="dashed", lw=2, alpha=0.5)
                ax1.plot(tx, exact_solution, color="black")
                ax1.plot(tx, pred_y, color="green")
                ax1.fill_between(np.squeeze(tx), np.squeeze((pred_y)-pred_s), np.squeeze((pred_y)+pred_s), color="green", alpha=0.3)
                ax1.fill_betweenx(np.array([-40, 30]), x_new[0]-r_x0[0], x_new[0]+r_x0[0], color="red", alpha=0.1) 
                ax1.set_xlim(0.2, 1.5)
                ax1.set_ylim(-35.0, 25.0)
                ax1.set_xlabel('x')
                ax1.set_ylabel('f(x)')
                ax1.fill_betweenx(np.array([-40, 30]), xopt1-mop.surr.radius[0], xopt1+mop.surr.radius[0], color="gray", alpha=0.25)
                ax1.set_ylim(-40, 30)
                ax1.set_xlim(0.2, 1.5)
                
                # Expected improvement
                ax2.plot(tx, ei_parallel_1, color="gray")
                ax2.plot(tx, ei_parallel_2, color="green")
                ax2.set_xlim(0.2, 1.5)
                ax2.axvline(x=xopt1, color="gray", ls="dashed", lw=2, alpha=0.5)
                ax2.axvline(x=xopt2, color="red", ls="dashed", lw=2, alpha=0.5)
                ax2.set_xlabel('x')
                ax2.set_ylabel('EI(x)')
                ax2.set_xlim(0.2, 1.5)
                
                # Penalisation on second axis
                ax3 = ax2.twinx()
                ax3.plot(tx, phi_failure, color="red")
                ax3.plot(tx, phi_parallel, color="blue")           
                ax3.fill_betweenx(np.array([-40, 30]), x_new[0]-r_x0[0], x_new[0]+r_x0[0], color="red", alpha=0.1)
                ax3.fill_betweenx(np.array([-40, 30]), xopt1-mop.surr.radius[0], xopt1+mop.surr.radius[0], color="gray", alpha=0.25)
                ax3.set_ylim(-0.1, 1.1)
                ax3.set_ylabel('phi(x)')
                ax3.set_ylim(-0.05, 1.05)
                
                # Save as png
                plt.pause(0.005)
                plt.show()
                plt.tight_layout()
                fig.savefig('Episode' + '_' + str(i-1) + '_' 'xopt2'+ '.png', transparent=False)            
            
            if(r_x0 < r_small):
                print('===================== xopt3: plotting  =====================')
                print("Plotting: radius is less than 1mm, no more samples are taken")
                print('============================================================')
            else:
        
                ###############
                # Third Sample:
                ###############
                
                # New is old
                mop.surr.xinds = x_new
                mop.surr.radius = r_x0
                mop.surr.std_dev_inds = s_x0
                
                
                hammer = mop.surr._hammer_function(tx, x_new, r_x0, s_x0)
                hammer = hammer[:, None]
                phi_parallel = phi_parallel*hammer
                mop.surr.phi_parallel = phi_parallel
                
                # store them
                if mop.surr.pen_locations is None:
                    mop.surr.pen_locations = np.atleast_2d(x_new)
                    mop.surr.r_x0 = np.array(r_x0)
                    mop.surr.s_x0 = np.array(s_x0)
                    
                else:
                    mop.surr.pen_locations = np.concatenate((mop.surr.pen_locations, np.atleast_2d(x_new)))
                    mop.surr.r_x0 = np.concatenate((mop.surr.r_x0, r_x0))
                    mop.surr.s_x0 = np.concatenate((mop.surr.s_x0, s_x0))

                # calculate hammer function for failures
                if not mop.surr.success:
                    N = len(mop.surr.global_failed_indices)
                    r_x0 = np.zeros(N)
                    s_x0 = np.zeros(N)
                    x0 = np.zeros((N, n_dim))
                    
                    for kkk in range(N):
                        x0[kkk] = np.array([mop.surr.global_failed_indices[0][kkk]])
                        r_x0[kkk], s_x0[kkk] = mop.surr._hammer_function_precompute(x0[kkk], np.amin(failure_gradient_weighting*mop.surr.L), mop.surr.best, mop.surr)
    
                    mop.surr.pen_locations = np.concatenate((mop.surr.pen_locations, x0))
                    mop.surr.r_x0 = np.concatenate((mop.surr.r_x0, r_x0))
                    mop.surr.s_x0 = np.concatenate((mop.surr.s_x0, s_x0))                
                
                # New penalisation
#                radius, std_dev_inds = mop.surr._hammer_function_precompute(x_new, parallel_gradient_weighting*mop.surr.L, mop.surr.best, mop.surr)
#                hammer = mop.surr._hammer_function(tx, x_new, radius, std_dev_inds)
#                hammer = hammer[:, None]
#                phi_parallel = phi_parallel*hammer
#                mop.surr.phi_parallel = phi_parallel
                
                # Penalise EI
                ei_parallel_3 = mop.surr.penalized_acquisition(tx, obj_sense=-1, lb=lb, ub=ub, cfunc=cfunc, cargs=cargs, ckwargs=ckwargs)
                             
                # Search for new optimimum location
                if n_dim > 1:
                    xopt3 = Optimiser.CMA_ES(toolbox, cma_centroid, cma_sigma, cma_options=cma_options)
                else:
                    xopt3 = Optimiser.grid_search_1D(toolbox, lb, ub, n=maxfevals)
                                
                # New sample point location from CMA-ES optimisation
                x_new = np.reshape(xopt3, (1, -1))
                
                print('===================== xopt3: plotting  =====================')
                
                r_x0, s_x0 = mop.surr._hammer_function_precompute(x_new, np.amin(parallel_gradient_weighting*mop.surr.L), mop.surr.best, mop.surr)
                # New values according to x_new prediction
#                y_new, std_new = mop.surr.predict(x_new)
#                r_new = (abs(mop.surr.best - y_new) / (parallel_gradient_weighting*mop.surr.L)) 
                
                # Print out new and old values
                jj += 1
                print("Plotting: Sample Two = ", jj)
                print("Plotting: x_old: ", mop.surr.xinds)
                print("Plotting: r_old: ", mop.surr.radius)
                print("Plotting: std_old: ", mop.surr.std_dev_inds)
                print("Plotting: x_new: ", x_new) 
                print("Plotting: r_new: ", r_x0)
                print("Plotting: std_new: ", s_x0)
                print("Plotting:  min(phi_parallel)", min(phi_parallel))
                print('============================================================')
                
                # include new sample in the training data.
                xtr = np.concatenate([xtr, x_new])
                
                ###################
                # Creation of plots
                ###################
                
                # Create figures
                fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))
                fig.subplots_adjust(hspace=0.25)
                y = Y.copy()
                xtmp = X.copy()
                
                # Surrogate model
                ax1.scatter(xtmp[:n_samples], y[:n_samples], marker="x", color="red", alpha=0.75)
                ax1.scatter(xtmp[n_samples:], y[n_samples:], c="red", alpha=0.75)
                ax1.axvline(x=xopt2, color="gray", ls="dashed", lw=2, alpha=0.5)
                ax1.axvline(x=xopt3, color="red", ls="dashed", lw=2, alpha=0.5)
                ax1.axhline(y=mop.surr.best, color="green", ls="dashed", lw=2, alpha=0.5)
                ax1.plot(tx, exact_solution, color="black")
                ax1.plot(tx, pred_y, color="green")
                ax1.fill_between(np.squeeze(tx), np.squeeze((pred_y)-pred_s), np.squeeze((pred_y)+pred_s), color="green", alpha=0.3)
                ax1.fill_betweenx(np.array([-40, 30]), x_new[0]-r_x0[0], x_new[0]+r_x0[0], color="red", alpha=0.1) 
                ax1.set_xlabel('x')
                ax1.set_ylabel('f(x)')
                ax1.fill_betweenx(np.array([-40, 30]), xopt2-mop.surr.radius[0], xopt2+mop.surr.radius[0], color="gray", alpha=0.25)
                ax1.set_ylim(-40, 30)
                ax1.set_xlim(0.2, 1.5)
                
                # Expected improvement
                ax2.plot(tx, ei_parallel_2, color="gray")
                ax2.plot(tx, ei_parallel_3, color="green")
                ax2.set_xlim(0.2, 1.5)
                ax2.axvline(x=xopt2, color="gray", ls="dashed", lw=2, alpha=0.5)
                ax2.axvline(x=xopt3, color="red", ls="dashed", lw=2, alpha=0.5)
                ax2.set_xlabel('x')
                ax2.set_ylabel('EI(x)')
                ax2.set_xlim(0.2, 1.5)
                
                # Penalisation on second axis
                ax3 = ax2.twinx()
                ax3.plot(tx, phi_failure, color="red")
                ax3.plot(tx, phi_parallel, color="blue")           
                ax3.fill_betweenx(np.array([-40, 30]), x_new[0]-r_x0[0], x_new[0]+r_x0[0], color="red", alpha=0.1)
                ax3.fill_betweenx(np.array([-40, 30]), xopt2-mop.surr.radius[0], xopt2+mop.surr.radius[0], color="gray", alpha=0.25)
                ax3.set_ylim(-0.1, 1.1)
                ax3.set_ylabel('phi(x)')
                ax3.set_ylim(-0.05, 1.05)
                
                # Save as png
                plt.pause(0.005)
                plt.show()
                plt.tight_layout()
                fig.savefig('Episode' + '_' + str(i-1) + '_' 'xopt3'+ '.png', transparent=False)          
                
                
            if(r_x0 < r_small):
                print('===================== xopt4: plotting  =====================')
                print("Plotting: radius is less than 1mm, no more samples are taken")
                print('============================================================')
                
            else:    
                
                ################
                # Fourth Sample:
                ################
                
                # New is old
                mop.surr.xinds = x_new
                mop.surr.radius = r_x0
                mop.surr.std_dev_inds = s_x0
                
                
                hammer = mop.surr._hammer_function(tx, x_new, r_x0, s_x0)
                hammer = hammer[:, None]
                phi_parallel = phi_parallel*hammer
                mop.surr.phi_parallel = phi_parallel
                
                # store them
                if mop.surr.pen_locations is None:
                    mop.surr.pen_locations = np.atleast_2d(x_new)
                    mop.surr.r_x0 = np.array(r_x0)
                    mop.surr.s_x0 = np.array(s_x0)
                    
                else:
                    mop.surr.pen_locations = np.concatenate((mop.surr.pen_locations, np.atleast_2d(x_new)))
                    mop.surr.r_x0 = np.concatenate((mop.surr.r_x0, r_x0))
                    mop.surr.s_x0 = np.concatenate((mop.surr.s_x0, s_x0))

                # calculate hammer function for failures
                if not mop.surr.success:
                    N = len(mop.surr.global_failed_indices)
                    r_x0 = np.zeros(N)
                    s_x0 = np.zeros(N)
                    x0 = np.zeros((N, n_dim))
                    
                    for kkk in range(N):
                        x0[kkk] = np.array([mop.surr.global_failed_indices[0][kkk]])
                        r_x0[kkk], s_x0[kkk] = mop.surr._hammer_function_precompute(x0[kkk], np.amin(failure_gradient_weighting*mop.surr.L), mop.surr.best, mop.surr)
    
                    mop.surr.pen_locations = np.concatenate((mop.surr.pen_locations, x0))
                    mop.surr.r_x0 = np.concatenate((mop.surr.r_x0, r_x0))
                    mop.surr.s_x0 = np.concatenate((mop.surr.s_x0, s_x0))  
                
                # Penalise EI
                ei_parallel_4 = mop.surr.penalized_acquisition(tx, obj_sense=-1, lb=lb, ub=ub, cfunc=cfunc, cargs=cargs, ckwargs=ckwargs)
                
                # Search for new optimimum location                     
                if n_dim > 1:
                    xopt4 = Optimiser.CMA_ES(toolbox, cma_centroid, cma_sigma, cma_options=cma_options)
                else:
                    xopt4 = Optimiser.grid_search_1D(toolbox, lb, ub, n=maxfevals)
                
                # New sample point location from CMA-ES optimisation
                x_new = np.reshape(xopt4, (1, -1))
                
                print('===================== xopt4: plotting  =====================')
                           
                # New values according to x_new prediction
                r_x0, s_x0 = mop.surr._hammer_function_precompute(x_new, np.amin(parallel_gradient_weighting*mop.surr.L), mop.surr.best, mop.surr)
#                y_new, std_new = mop.surr.predict(x_new)
#                r_new = (abs(mop.surr.best - y_new) / (parallel_gradient_weighting*mop.surr.L)) 
                
                # Print out new and old values
                jj += 1
                print("Plotting: Sample Two = ", jj)
                print("Plotting: x_old: ", mop.surr.xinds)
                print("Plotting: r_old: ", mop.surr.radius)
                print("Plotting: std_old: ", mop.surr.std_dev_inds)
                print("Plotting: x_new: ", x_new)
                print("Plotting: r_new: ", r_x0)
                print("Plotting: std_new: ", s_x0)
                print("Plotting:  min(phi_parallel)", min(phi_parallel))
                print('============================================================')
                
                # include new sample in the training data.
                xtr = np.concatenate([xtr, x_new])
                
                ###################
                # Creation of plots
                ###################
                
                # Create figures                
                fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))
                fig.subplots_adjust(hspace=0.25)
                y = Y.copy()
                xtmp = X.copy()
                
                # Surrogate model
                ax1.scatter(xtmp[:n_samples], y[:n_samples], marker="x", color="red", alpha=0.75)
                ax1.scatter(xtmp[n_samples:], y[n_samples:], c="red", alpha=0.75)
                ax1.axvline(x=xopt3, color="gray", ls="dashed", lw=2, alpha=0.5)
                ax1.axvline(x=xopt4, color="red", ls="dashed", lw=2, alpha=0.5)
                ax1.axhline(y=mop.surr.best, color="green", ls="dashed", lw=2, alpha=0.5)
                ax1.plot(tx, exact_solution, color="black")
                ax1.plot(tx, pred_y, color="green")
                ax1.fill_between(np.squeeze(tx), np.squeeze((pred_y)-pred_s), np.squeeze((pred_y)+pred_s), color="green", alpha=0.3)
                ax1.fill_betweenx(np.array([-40, 30]), x_new[0]-r_x0[0], x_new[0]+r_x0[0], color="red", alpha=0.1) 
                ax1.set_xlabel('x')
                ax1.set_ylabel('f(x)')
                ax1.fill_betweenx(np.array([-40, 30]), xopt3-mop.surr.radius[0], xopt3+mop.surr.radius[0], color="gray", alpha=0.25)
                ax1.set_ylim(-40, 30)
                ax1.set_xlim(0.2, 1.5)
                
                # Expected improvement
                ax2.plot(tx, ei_parallel_3, color="gray")
                ax2.plot(tx, ei_parallel_4, color="green")
                ax2.set_xlim(0.2, 1.5)
                ax2.axvline(x=xopt3, color="gray", ls="dashed", lw=2, alpha=0.5)
                ax2.axvline(x=xopt4, color="red", ls="dashed", lw=2, alpha=0.5)
                ax2.set_xlabel('x')
                ax2.set_ylabel('EI(x)')
                ax2.set_xlim(0.2, 1.5)
                
                # Penalisation on second axis
                ax3 = ax2.twinx()
                ax3.plot(tx, phi_failure, color="red")
                ax3.plot(tx, phi_parallel, color="blue")           
                ax3.fill_betweenx(np.array([-40, 30]), x_new[0]-r_x0[0], x_new[0]+r_x0[0], color="red", alpha=0.1)
                ax3.fill_betweenx(np.array([-40, 30]), xopt3-mop.surr.radius[0], xopt3+mop.surr.radius[0], color="gray", alpha=0.25)
                ax3.set_ylim(-0.1, 1.1)
                ax3.set_ylabel('phi(x)')
                ax3.set_ylim(-0.05, 1.05)
                
                # Save as png
                plt.pause(0.005)
                plt.show()
                plt.tight_layout()
                fig.savefig('Episode' + '_' + str(i-1) + '_' 'xopt4'+ '.png', transparent=False)    
                        
            
            if verbose:
                print ("=======================")
            if n_obj > 1:
                print ('Hypervolume: ', hpv[-1])   
            else:
                print ('Best function value: ', hpv[-1])
            if i%1 == 0:
                print('Saving data...')
                try:
                    np.savez(sim_file, X, Y, hpv, (time.time()-start_sim)/60.0)
                    print('Data saved in file: ', sim_file)
                except Exception as e:
                    print(e)
                    print('Data saving failed.')
            
                  
        print('Saving data...')
        try:
            np.savez(sim_file, X, Y, hpv, (time.time()-start_sim)/60.0)
            print('Data saved in file: ', sim_file)
        except Exception as e:
            print(e)
            print('Data saving failed.')
            
        print("EMO(): Data saved")
        return X, Y, hpv
        
