#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:31:00 2019

@author: andrew
"""

import numpy as np
from pyDOE import lhs as LHS
import subprocess
import os
import time
from scipy.special import erf as ERF
#import sys
from os.path import expanduser

import IscaOpt 

# home directory
home = expanduser("~")

from hydro_plane import Ellipse

from multiprocessing import Process, Queue, current_process, freeze_support

try:
    from data import support
except:
    from .data import support
    
#from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

current = os.getcwd()   

from scipy.spatial.distance import euclidean
import shutil


# APR added
try:
    from data.SnappyHexOptimise import BasicHeatExchangerRun
except:
    from .data.SnappyHexOptimise import BasicHeatExchangerRun
try:
    from interfaces import EllipseInterface
except:
    from .interfaces import EllipseInterface
try:
    from base_class import Problem
except:
    from .base_class import Problem
try:
    from data import support
except:
    from .data import support

#############
# Environment
############# 
# On ISCA, there can be batches of up to the maximum number of samples

#environment = 'isambard'
#environment = 'isambard_test'
#environment = 'isca'
#environment = 'isca_test'

#environment = 'isca_forrester'
environment = 'local_forrester'

no_of_nodes = 4

# Locally, there can only be three "nodes" as each one has 2 CPUS, and I only have 8 CPUs
#environment = 'local'
#no_of_nodes = 3

##########
# Sampling
##########
#sampling = 'manual'
#sampling = 'latin'
sampling = 'latin_forrester'

##########################
# Selection of bash script
##########################

# this bash script is for the OpenFOAM run
if environment == 'isca':    
    bash = 'run_script_isca_parallel.sh'
elif environment == 'isca_test':
    # initial sampling does not need to use the optimiser!!!!
    bash = 'isca_initial_sampling_parallel_test.sh'     
elif environment == 'isambard':
    # initial sampling does not need to use the optimiser!!!!
    bash = 'isambard_initial_sampling_parallel.sh'
elif environment == 'isambard_test':
    # initial sampling does not need to use the optimiser!!!!
    bash = 'isambard_initial_sampling_parallel_test.sh'    
elif environment == 'isca_forrester':
    bash = 'isca_forrester_initial_sampling_parallel.sh'    
else:
    bash = 'local_forrester_initial_sampling_parallel.sh'    

################
# Create failure
################
failure = True    
    
# HeadCell object:
class HeadCell(Problem, EllipseInterface):

    def __init__(self, settings):
        self.source_case = settings.get('source_case', './data/HeadCell/source/')            
        self.case_path = settings.get('case_path', './data/HeadCell/case_local/')
        self.mesh_path = settings.get('mesh_path', './data/HeadCell/meshes/')
        self.setup()
        

    def setup(self, verbose=False):
        """
        Just sets values
        """
        self.L = 4.0
        self.A = 45.0
        self.R = 1.0
        self.xlb, self.xub = -self.L, self.L
        self.zlb, self.zub = -self.L, self.L
        self.anglelb, self.angleub = 0, self.A
        self.majorlb, self.majorub = 4.0*self.R, 8.0*self.R 
        self.minorlb, self.minorub = 4.0*self.R, 8.0*self.R
        
        EllipseInterface.__init__(self, self.xlb, self.xub, self.zlb, self.zub, \
                                 self.anglelb, self.angleub, self.majorlb, self.majorub, \
                                 self.minorlb, self.minorub)

    def info(self):
        raise NotImplementedError

    def get_configurable_settings(self):
        raise NotImplementedError

    def run(self, shape, verbose=False):
        xp, yp, rp = shape
        support.circle_to_stl(rp, xp, yp, \
            file_directory=self.case_path+self.stl_dir, file_name=self.stl_file_name, draw=False)
        t, p = self.problem.cost_function(sense="multi", verbose=verbose)
        return t, p

    def evaluate(self, decision_vector, verbose=False):
        if not self.constraint(decision_vector):
            raise ValueError('Constraint violated. Please supply a feasible decision vector.')
        shape = self.convert_decision_to_shape(decision_vector)
        try:
            return self.run(shape, verbose)
        except Exception as e:
            print('Solution evaluation failed.')
            print(e)

def lhs_initial_samples(n_dim, ub, lb, n_samples=4, cfunc=None, cargs=(), ckwargs={}):
    """
    Generate Latin hypercube samples from the decision space using pyDOE.

    Parameters.
    -----------
    n_samples (int): the number of samples to take. 
    cfunc (method): a cheap constraint function.
    cargs (tuple): arguments for cheap constraint function.
    ckawargs (dictionary): keyword arguments for cheap constraint function. 

    Returns a set of decision vectors.         
    """
    seed = 1234
    np.random.seed(seed)
    samples = LHS(n_dim, samples=n_samples)
        
    scaled_samples = ((ub - lb) * samples) + lb            
                        
    if cfunc is not None: # check for constraints
        print('Checking for constraints.')
        scaled_samples = np.array([i for i in scaled_samples if cfunc(i, *cargs, **ckwargs)])

    return scaled_samples

# constaints:
def gap_and_checkMesh_constraint(x, layout):
    """
    Create meshes and check the constraints using the decision vector and layout.

    Parameters.
    -----------
    x (numpy array): the decision vector. 
    layout (Ellipse object): the object for the generation of the pointwise files. 

    Returns whether the constraint was successfully passed.         
    """
    # get the current directory
    current = os.getcwd()
    
    # defaults for success:
    minimum_gap_success = True
    checkMesh_success = False
    success = False
    
    # checkMesh:
    utility3 = "checkMesh"
    
    # platform specific changes (not used by Isambard):
    if environment == 'blades':
        Pointwise_path='/usr/local/Pointwise/PointwiseV18.0R2/pointwise'
    elif environment == 'local':
        Pointwise_path='/home/andrew/Pointwise/PointwiseV18.0R2/pointwise'
    elif ((environment == 'isca') or (environment == 'isca_test')):    
        Pointwise_path = '/gpfs/ts0/home/apr207/Pointwise/PointwiseV18.0R2/pointwise'
    else:
        #local is default
        Pointwise_path='/home/andrew/Pointwise/PointwiseV18.0R2/pointwise'
    
    # paths:
    mesh_path = "/data/HeadCell/meshes/"
    source_path = "/data/HeadCell/source/"
    
    # the neccessary name for the child directory:
    dir_name = "_"
    
    for j in range(len(x)):
        dir_name += "{0:.4f}_".format(x[j])
        dir_name = dir_name.replace("[","")
        dir_name = dir_name.replace("]","")

    # make the child directory:
    subprocess.call(['mkdir', '-p', current + mesh_path + dir_name + '/system' ])
    subprocess.call(['mkdir', '-p', current + mesh_path + dir_name + '/constant/polyMesh' ])

    # location of centre of ellipse:
    bottom_x_centre = x[0]
    bottom_z_centre = x[1]
    
    # rotation angle:
    rot_angle = x[2]*(np.pi/180.0)
    
    # ellipse sizes:
    a = x[3]
    b = x[4]
    
    # create space for corners and gaps:
    corners = np.zeros((4,2))
    gaps = np.zeros(4)
    
    # the bottom of the tray:
    y_bottom = -14.9375
    
    # very top of tray: 
    p1 = np.array([0, 6.0625, 0])
    
    # very bottom of tray:
    p2 = np.array([bottom_x_centre, y_bottom, bottom_z_centre])
    
    # the radii at the first tray layer:
    a_top = layout.get_ellipse_radii(21.25, a, -2.34375)
    b_top = layout.get_ellipse_radii(21.25, b, -2.34375)
    
    # the points at the first tray layer:
    c_top = layout.get_ellipse_points(p1, p2, -2.34375)
    
    # the angle at the first tray layer:
    angle_top = layout.get_ellipse_angles(rot_angle, -2.34375)
    
    # obtain the four corners:
    for i in range(1,5):
        angles = np.linspace(start=0.5*i*np.pi, stop=0.5*(i-1)*np.pi, num=1, endpoint=True)
        corners[i-1,:] = layout.generate_ellipse(c_top, a_top, b_top, angles, theta=angle_top+np.pi, n=0)

    # compute horizontal gaps:
    for j in range(0,4):
        gaps[j] = 21.25 - np.sqrt((corners[j,0])**2 + (corners[j,1])**2)
    
    # minimim horizontal gap:
    min_gap=min(gaps)

    # get the index of the minimum horizontal gap and the maximum x-extent:
    index = np.where(gaps == min_gap)
    x_extent_vector = abs(corners[index])
    x_extent= np.amax(x_extent_vector)
    
    # Definition of the x-y plane locations
    x1 = 21.25
    x2 = x_extent
    x3 = x_extent
    x4 = 21.25
    
    y1 = -2.9375
    y2 = -2.34375
    y3 = -6.59375
    y4 = 1.3125
    
    # Hero's formula for the normal gap
    A = y2 - y3
    B = np.sqrt((y4 - y2)**2 + (x4 - x2)**2)
    P = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    S = (A+B+P)/2
    Area = np.sqrt(S*(S-A)*(S-B)*(S-P))
    H = 2*Area/B
    
    
    
    # constraints:
    if (H < 2.0):
        # is the centre in the right place?:
        minimum_gap_success = False
        
    else:

        # copy controlDict and fvSchemes ssfor checkMesh 
        shutil.copyfile(current + source_path + 'system/controlDict', current + mesh_path + dir_name + '/system/controlDict')
        shutil.copyfile(current + source_path + 'system/fvSchemes', current + mesh_path + dir_name + '/system/fvSchemes')
        shutil.copyfile(current + source_path + 'system/fvSolution', current + mesh_path + dir_name + '/system/fvSolution')
        
        # copy all mesh files to mesh path
        for filename in os.listdir(current + source_path):
            full_file_name = os.path.join(current + source_path, filename)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, current + mesh_path + dir_name)
                
        # update Pointwise files using decision vector (needs checking)
        layout.update(x)
        
        # decide where we are to run Pointwise:
        if ((environment == 'isambard') or (environment == 'isambard_test')):
                
            # copy pointwise_script.sh to submit script
            shutil.copyfile(current + '/pointwise_script.sh', current + mesh_path + dir_name + '/pointwise_script.sh')
            
            # write the location of the current directory to a shell script - used by ssh and qsub
            with open(home + "/pointwise_directory.sh", "a") as myfile:
                myfile.write('export POINTWISE_DIRECTORY=' + '"'+ current + mesh_path + dir_name + '"' + '\n')
            
            print("checkMesh_constraint(): written current directory to a file")
            
            # name of the script to submit pointwise with
            bash_pointwise = 'pointwise_script.sh'
            
            # change bash_pointwise in order to run from current directory
            with open(current + mesh_path + dir_name + '/' + bash_pointwise, 'r') as f:
                data = f.readlines()
            
            # read in each line and change the directory location                
            for line in range(len(data)):       
                if '#PBS -o' in data[line]:
                    data[line] = '#PBS -o 10.141.0.1:' + current + mesh_path + dir_name + '/log.pointwise' + '\n'
            
                if '#PBS -e' in data[line]:                                                                                                                                                 
                    data[line] = '#PBS -e 10.141.0.1:' + current + mesh_path + dir_name + '/err.pointwise' + '\n'
                    
            # write the changes to bash_pointwise        
            with open(current + mesh_path + dir_name + '/' + bash_pointwise, 'w') as f:
                f.writelines(data)
            
            # use ssh to send the job to phase 1
            subprocess.call(['ssh','ex-aroberts@login-01.gw4.metoffice.gov.uk', 'source $HOME/pointwise_directory.sh; export PBS_HOME=/cm/shared/apps/pbspro/var/spool; export PBS_EXEC=/cm/shared/apps/pbspro/19.2.4.20190830141245; /cm/shared/apps/pbspro/19.2.4.20190830141245/bin/qsub $POINTWISE_DIRECTORY/pointwise_script.sh;'], cwd=current)
            
            # check if the log files exist
            while not (os.path.exists(current + mesh_path + dir_name + '/' + 'log.pointwise') and os.path.exists(current + mesh_path + dir_name + '/' + 'err.pointwise')):
                time.sleep(1)
                
            print("checkMesh_constraint(): pointwise completed")
            
            # remove the shell script from the home directory
            subprocess.call(['rm', home + '/' + 'pointwise_directory.sh'])
            
    
        else:    
            # run pointwise on isca or locally using Pointwise_path:
            subprocess.call([Pointwise_path, '-b', current + mesh_path + dir_name + '/Hydro_V18_3_tray_APR_grit_pot_parameterised_ellipse_correction.glf'], cwd=current, \
                              stdout = open(current + mesh_path + dir_name + '/log.pointwise', 'w'), \
                              stderr = open(current + mesh_path + dir_name + '/err.pointwise', 'w'))
        
        # run checkmesh:
        subprocess.call([utility3, '-case', current + mesh_path + dir_name], cwd=current, \
                      stdout = open(current + mesh_path + dir_name + '/log.checkMesh', 'w'), \
                      stderr = open(current + mesh_path + dir_name + '/err.checkMesh', 'w'))
              

        
        # check if the mesh was successful and assign boolean if succeeded              
        with open(current + mesh_path + dir_name + "/log.checkMesh") as f:
            if 'Mesh OK.' in f.read():
                checkMesh_success = True


    # Convert boolean into integer
    if minimum_gap_success == True:
        g_success = 1
    else:
        g_success = 0

                 
    # Convert boolean into integer
    if checkMesh_success == True:
        c_success = 1
    else:
        c_success = 0

    # Load the constraints file
    file_constraints = 'constraints.npz'
    data = np.load(current + '/' + file_constraints)

    # Load and set the decision vector
    X = data['arr_0']
    X_current = x

    if X.size != 0:
        X_new = np.vstack((X, X_current))
        print('appended ', X_current, ' to ', X)
        print('this gives ', X_new)
    else:
        X_new = X_current
        print('set ', X_new, ' to ', X_current)


    # Load and set gap constaint success
    G = data['arr_1']
    G_current = g_success

    if G.size != 0:
        G_new = np.vstack((G, G_current))
        print('appended ', G_current, ' to ', G)
        print('this gives ', G_new)
    else:
        G_new = G_current
        print('set ', G_new, ' to ', G_current)


    # Load and set checkMesh success
    C = data['arr_2']
    C_current = c_success

    if C.size != 0:
        C_new = np.vstack((C, C_current))
        print('appended ', C_current, ' to ', C)
        print('this gives ', C_new)
    else:
        C_new = C_current
        print('set ', C_new, ' to ', C_current)




    # Save the decision vector and checkMesh to a file:
    try:     
        np.savez(current + '/' + file_constraints, X_new, G_new, C_new)
        print('Data saved in file: ', file_constraints, ' X ', X_new, ' gap constraint ', G_new, ' checkMesh ', C_new)
    except Exception as e:
        print(e)
        print('Data saving failed.')                  
                
                
                

#    try:
#        np.savez(current + '/' + file_C, initial_X, initial_C)
#        print('Data saved in file: ', file_C)
#    except Exception as e:
#        print(e)
#        print('Data saving failed.') 


                
            
            
                
        
    print('gap_and_checkMesh_constraint(x): minimum_gap_success passed? ', str(minimum_gap_success), ' as H = ', H)
    print('gap_and_checkMesh_constraint(x): checkMesh_success passed? ', str(checkMesh_success))
            
    success = checkMesh_success and minimum_gap_success
    
    # if either it failed the centre location check or the checkMesh, delete the mesh:
    if(checkMesh_success == False):
        # remove mesh if there is a mesh failure
        subprocess.call(['rm', '-r', current + mesh_path + dir_name + '/'])
        print('centre_constraint(): mesh deleted')

    return success

#     
# NOT USED FOR INITIAL ISAMBARD TEST:
#    
def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = calculate(func, args)
        output.put(result)

#     
# NOT USED FOR INITIAL ISAMBARD TEST:
#   
def calculate(func, args):
    result = func(*args)
    return '%s says that %s%s = %s' % \
        (current_process().name, func.__name__, args, result)

#     
# NOT USED FOR INITIAL ISAMBARD TEST:
#   
def mul(d):
    
    if ((environment == 'isca') or (environment == 'isca_test') or (environment == 'isca_forrester')):
        start_msub = time.time()
        subprocess.call(['msub', '-K', d + '/' + bash], cwd=d, \
              stdout = open(d + '/log.subprocess', 'w'), \
              stderr = open(d + '/err.subprocess', 'w'))

    # submit job using qsub if on isambard:
    elif ((environment == 'isambard') or (environment == 'isambard_test')):
        start_msub = time.time()
        subprocess.call(['qsub', '-W', 'block=true', d + '/' + bash], cwd=d, \
              stdout = open(d + '/log.subprocess', 'w'), \
              stderr = open(d + '/err.subprocess', 'w'))

        
    else:
        start_msub = time.time()
        subprocess.call(['sh', d + '/' + bash], cwd=d, \
              stdout = open(d + '/log.subprocess', 'w'), \
              stderr = open(d + '/err.subprocess', 'w'))
        
    return "Done time," + str(time.time() - start_msub)

#     
# NOT USED FOR INITIAL ISAMBARD TEST:
#   
def queue():
    
    filedirs = initialisation()
            
    task_list = [(mul, (d,)) for d in filedirs]    
    
    # Create queues
    task_queue = Queue()
    done_queue = Queue()
    
    # Submit tasks
    for task in task_list:
        task_queue.put(task)

    # Start worker processes
    for i in range(no_of_nodes):
        Process(target=worker, args=(task_queue, done_queue)).start()    
    
    # Get and print results
    print('Unordered results:')
    for i in range(len(task_list)):
        print('\t', done_queue.get())
    
    # Tell child processes to stop
    for i in range(no_of_nodes):
        task_queue.put('STOP')

def no_queue():
    """
    Submit the job to the relevant queue without a python queue.

    Parameters.
    -----------
    There are no parameters. 

    Does not return anything.         
    """
    filedirs = initialisation()
    
    # submit job using moab if on isca:
    if ((environment == 'isca') or (environment == 'isca_test') or (environment == 'isca_forrester')):
        for d in range(len(filedirs)):
            subprocess.Popen(['msub', filedirs[d] + '/' + bash], cwd=filedirs[d], \
                  stdout = open(filedirs[d] + '/log.subprocess', 'w'), \
                  stderr = open(filedirs[d] + '/err.subprocess', 'w'))
    
    # submit job using qsub if on isambard:
    elif ((environment == 'isambard') or (environment == 'isambard_test')):
        for d in range(len(filedirs)):
            subprocess.Popen(['qsub', filedirs[d] + '/' + bash], cwd=filedirs[d], \
                  stdout = open(filedirs[d] + '/log.subprocess', 'w'), \
                  stderr = open(filedirs[d] + '/err.subprocess', 'w'))
    
    # submit job as a shell script if on local pc:
    else:
        for d in range(len(filedirs)):
            subprocess.Popen(['sh', filedirs[d] + '/' + bash], cwd=filedirs[d], \
                  stdout = open(filedirs[d] + '/log.subprocess', 'w'), \
                  stderr = open(filedirs[d] + '/err.subprocess', 'w'))


def initialisation():
    """
    Samples the design space,
    Writes the decision vector to a file
    Creates an empty initial samples file
    Copies and edits the qsub or msub submission script.

    Parameters.
    -----------
    There are no parameters. 

    Returns a list of directories for submission.         
    """
    
    
    if(sampling == 'latin_forrester'):
        # number of dimensions and samples    
        n_dim = 1
        #n_samples = 11*n_dim -1
        n_samples = 5
        ub = 1.4
        lb = 0.3
        print("initialisation(): number of Latin Hypercube samples", str(n_samples))
        samples = lhs_initial_samples(n_dim, ub, lb, n_samples, cfunc=None, cargs=(), ckwargs={})
        samples = np.array([np.array([0.5501084598698482]), np.array([0.6004338394793927]), np.array([0.9509761388286335]), np.array([1.2008676789587853]), np.array([1.3015184381778744])])
        
        case_path = "/data/Forrester/case_local/"
        source_file = 'run_case.py'
        
        # create filedirs
        filedirs = []
        
        # loop through the list to create directories:
        for s in samples:
            
            # create a working directory from the sample:
            dir_name = "_"
            for j in range(len(s)):
                dir_name += "{0:.4f}_".format(s[j])
            
            # replace any directories containing []
            dir_name = dir_name.replace("[","")
            dir_name = dir_name.replace("]","")
            
            # add the name to a list of directories
            filedirs.append(current + case_path + dir_name)
            
            # create the directory from the last in the list and 
            subprocess.call(['mkdir', filedirs[-1] + '/'])
            
            # copy run_case.py to that directory
            subprocess.call(['cp', '-r', current + '/' + source_file, filedirs[-1] + '/'])
               
            # write the decision vector to a file        
            with open(filedirs[-1] + "/decision_vector.txt", "a") as myfile:
                for i in range(0,len(s)):
                    myfile.write(str(s[i])+ '\n')
            print("initialisation(): written decision vector to a file")
        
        
        # read decision vector and write to npz file
        print('Writing empty initial_samples.npz file...')
        
        # hyper volume improvement is null
        hpv = []
        initial_time = 0
        initial_X = []
        initial_Y = []

        
        # the name of the sim_file is initial_samples.npz    
        sim_file = 'initial_samples.npz'
        
        # remove npz file if it exists
        subprocess.call(['rm', '-r', current + '/' + sim_file]) 
        
        # initial_X is decision vector
        # initial_Y is 2 objectives
        # hpv is hypervolume improvement
        # initial_time is zero
        # initial_convergence is convergence failure  
        try:
            np.savez(current + '/' + sim_file, initial_X, initial_Y, hpv, initial_time)
            print('Data saved in file: ', sim_file)
        except Exception as e:
            print(e)
            print('Data saving failed.')

    
    else:
    
        # set the location of the case path
        case_path = "/data/HeadCell/case_local/"
        source_path = "/data/HeadCell/source/."
            
        source_file = 'run_case.py'
        
        # remove the current case path and create a new case path
        subprocess.call(['rm', '-r', current + case_path])    
        subprocess.call(['mkdir','-p', current + case_path])
        
        # get the lower and upper bounds
        lb, ub = prob.get_decision_boundary() 
        
        print("lb, ub", lb, ub)
    
        # number of dimensions and samples    
        n_dim = 5
        n_samples = 11*n_dim -1
        
        
        
        # read decision vector and write to checkMesh.npz file
        print('Writing empty constraints.npz file...')
        
        # gap and checkMesh is null
        initial_X = []
        initial_H = []
        initial_C = []
            
        file_constraints = 'constraints.npz'
        
        subprocess.call(['rm', '-r', current + '/' + file_constraints])
        
        # initial_X is decision vector
        # initial_H is gap constraint
        # initial_C is checkMesh
    
        try:
            np.savez(current + '/' + file_constraints, initial_X, initial_H, initial_C)
            print('Data saved in file: ', file_constraints)
        except Exception as e:
            print(e)
            print('Data saving failed.') 
        
        
        
        # take samples for the decision vector, either manual or latin hypercube
        if sampling == 'manual':
            print("initialisation(): number of manual samples", str(9))
            
            # 3 deliberate passes:
            one = np.array([0.5, 0.5, 10.0, 4.0, 4.0]) # 5,145,018 cells, 906 severly orthoginal faces, passes checkMesh Should result in H=3.02 inches
            two = np.array([1.0, 1.0, 20.0, 4.0, 5.0]) # 5,095,425 cells, 1087 severly orthoginal faces, passes checkMesh Should result in H=2.85 inches
            three = np.array([1.5, 1.5, 30.0, 4.0, 6.0]) # 4,993,048 cells, 1331 severly orthoginal faces, passes checkMesh Should result in H=2.67 inches
    
            # 3 deliberate failures
            four = np.array([4.0, 4.0, 10.0, 8.0, 6.0]) # Should result in H=1.51 inches
            five = np.array([4.0, -4.0, 30.0, 8.0, 4.0]) # Should result in H=1.62 inches
            six = np.array([4.0, -4.0, 45.0, 8.0, 4.0]) # Should result in H=1.76 inches
            
            # Mix of success and failure
            seven = np.array([1.0, 1.0, 10.0, 4.0, 5.0]) # Should result in H=2.83 inches
            eight = np.array([1.5, 1.5, 10.0, 4.0, 6.0]) # Should result in H=2.61 inches
            nine = np.array([1.0, 1.0, 30.0, 4.0, 5.0]) # Should result in H=2.88 inches
          
            all_samples = np.array([one, two, three, four, five, six, seven, eight, nine])
                    
            samples = []
            
            for x in all_samples:
                constraint_success = gap_and_checkMesh_constraint(x, layout)
                if constraint_success == True:
                    samples.append(x)
        else:
            print("initialisation(): number of Latin Hypercube samples", str(n_samples))
            samples = lhs_initial_samples(n_dim, ub, lb, n_samples, cfunc=gap_and_checkMesh_constraint, cargs=(layout,), ckwargs={})
        
        # create filedirs
        filedirs = []
        
        # loop through the list to create directories:
        for s in samples:
            
            # create a working directory from the sample:
            dir_name = "_"
            for j in range(len(s)):
                dir_name += "{0:.4f}_".format(s[j])
            
            # replace any directories containing []
            dir_name = dir_name.replace("[","")
            dir_name = dir_name.replace("]","")
            
            # add the name to a list of directories
            filedirs.append(current + case_path + dir_name)
            
            # create the directory from the last in the list and 
            subprocess.call(['mkdir', filedirs[-1] + '/'])
            
            # copy all source files into the newly created directory
            subprocess.call(['cp', '-r', current + source_path, filedirs[-1]])
            
            # copy run_case.py to that directory
            subprocess.call(['cp', '-r', current + '/' + source_file, filedirs[-1] + '/'])
               
            # write the decision vector to a file        
            with open(filedirs[-1] + "/decision_vector.txt", "a") as myfile:
                for i in range(0,len(s)):
                    myfile.write(str(s[i])+ '\n')
            print("initialisation(): written decision vector to a file")
        
        
        # read decision vector and write to npz file
        print('Writing empty initial_samples.npz file...')
        
        # hyper volume improvement is null
        hpv = []
        initial_time = 0
        initial_X = []
        initial_Y = []
        initial_convergence = []
        
        # the name of the sim_file is initial_samples.npz    
        sim_file = 'initial_samples.npz'
        
        # remove npz file if it exists
        subprocess.call(['rm', '-r', current + '/' + sim_file]) 
        
        # initial_X is decision vector
        # initial_Y is 2 objectives
        # hpv is hypervolume improvement
        # initial_time is zero
        # initial_convergence is convergence failure  
        try:
            np.savez(current + '/' + sim_file, initial_X, initial_Y, hpv, initial_time, initial_convergence)
            print('Data saved in file: ', sim_file)
        except Exception as e:
            print(e)
            print('Data saving failed.')













    # decide environment and adjust bash script for running.
    if ((environment == 'isca') or (environment == 'isca_test')):
        # if the environment is isca
        for d in range(len(filedirs)):
            
            # copy the bash run script to the run directory
            subprocess.call(['cp', current + '/' + bash, filedirs[d] + '/'])
                        
            # open the bash script for running
            with open(filedirs[d] + '/' + bash, 'r') as f:
                data = f.readlines()
                
            # change the line to the correct directory
            for line in range(len(data)):       
                if '#PBS -d' in data[line]:
                    data[line] = '#PBS -d '+ filedirs[d]+'/' + '\n'
            
            #write the lines to the file        
            with open(filedirs[d] + '/' + bash, 'w') as f:
                f.writelines(data)
                
    elif ((environment == 'isambard') or (environment == 'isambard_test')):
        # if the environment is isambard
        for d in range(len(filedirs)):
            
            # copy the bash run script to the run directory
            subprocess.call(['cp', current + '/' + bash, filedirs[d] + '/'])
                        
            # open the bash script for running
            with open(filedirs[d] + '/' + bash, 'r') as f:
                data = f.readlines()
                
            # change the line to the correct directory
            for line in range(len(data)):       
                if '#PBS -d' in data[line]:
                    data[line] = '#PBS -d '+ filedirs[d]+'/' + '\n'
            
            #write the lines to the file        
            with open(filedirs[d] + '/' + bash, 'w') as f:
                f.writelines(data)
                
    else:
        # if the environment is local:    
        for d in range(len(filedirs)):
            # copy the bash script to the local directory
            subprocess.call(['cp', current + '/' + bash, filedirs[d] + '/'])
            
    print('Total number of simulations, ', len(filedirs))
    print('All directories created.')

    return filedirs


def transform_sequence(n, lb=0, ub=185, scale=0.01, lw=0.25, uw=0.75):
    np = (n/ub * (ub -lb)) - ((ub -lb)/2)
    nub = (ub -lb)/2
    nlb = -(ub-lb)/2
    w = (ERF(scale*np) - ERF(scale*nlb))/(ERF(scale*nub) - ERF(scale*nlb))
    w = (w * (uw - lw)) + lw
    return w


def expected_improvement_weighting(self, x, obj_sense=1, lb=None, ub=None, weight=None, n=None,\
                            cfunc=None, cargs=(), ckwargs={}):
    """Calculate the expected improvement at a given set of input parameters,
    based on the trained model. See the following for details. 
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.139.9315&rep=rep1&type=pdf (maximisation)
    http://www.schonlau.net/publication/_96jsm_global_optimization.pdf (minimisation)
    
    Parameters.
    -----------
    x: a set of input parameters. 
    obj_sense: whether to maximise or minimise. Key for the input:
                 1: maximise
                -1: minimise (default)
    Returns the expected improvement value at x.
    """
    #print(x, x.shape)
    if len(x.shape) < 2:
        x = np.array(x)
        x = x[:, np.newaxis].T
    if lb is not None:
        #print(x, lb, ub)
        xp = np.clip(x.copy(), lb, ub)
        rows = x.shape[0]
        e = np.ones(rows)
        e_inds = [i for i in range(xp.shape[0]) \
            if np.any(np.all(np.abs(xp[i] - self.X) <= 1e-9, axis=1))]
        if len(e_inds) == rows:
            return e[:, np.newaxis].T
    c_inds = []
    if cfunc is not None:
        #print('==========')
        #print('xp: ', xp)
        if xp.shape[0] == 1:
            #print(cfunc(None, xp[0], *cargs, **ckwargs))
            if not cfunc(xp[0], *cargs, **ckwargs):
                #print('Infeasible solution.')
                return np.zeros((1,1))
        else:
            #print('------')
            #for i in xp:
            #    print(cfunc(None, i, *cargs, *ckwargs))
            #c_inds = np.array([i for i in range(xp.shape[0]) if not cfunc(xp[i], *cargs, **ckwargs)])
            xp = np.array([i for i in xp if cfunc(i, *cargs, **ckwargs)])
            
    y, std_dev = self.predict(xp) 
    epsilon = 0
    if n is not None:
        weight = transform_sequence(n)
    f_best = obj_sense * np.max(obj_sense * self.Y) + (obj_sense * epsilon)
    u =  obj_sense * (y - f_best) / std_dev
    # normal cumulative distribution function
    PHI = (0.5 * ERF(u/np.sqrt(2.0)))+0.5 
    # normal density function
    phi = 1/np.sqrt(2.0*np.pi)*np.exp(-u**2/2.0) 
    if weight is None:
        a =  std_dev * ((u * PHI) + (phi)) 
    else:       
        a =  std_dev * ((weight * u * PHI) + ((1 - weight) * phi))
    a[e_inds] = 0
    #a[c_inds] = 0
    #print(a.shape)
    return a

def expected_improvement(self, x, obj_sense=1, lb=None, ub=None,\
                            cfunc=None, cargs=(), ckwargs={}):
    """
    Calculate the expected improvement at a given set of decision vectors,
    based on the trained model. See the paper for details.
    
    Parameters.
    -----------
    x (np.array): a set of decision vectors.
    obj_sense (int): whether to maximise or minimise. Key for the input:
                 1: maximise
                -1: minimise (default)
    lb (np.array): lower bound for the decision space.
    ub (np.array): upper cound for the decision space.
    cfunc (function): cheap constraint fucntion.
    cargs (tuple): arguments for the constraint function.
    ckwargs (dict): keyword arguments for the constraint function.
    
    Returns the expected improvement values at x.
    """
    if len(x.shape) < 2:
        x = np.array(x)
        x = x[:, np.newaxis].T
    if lb is not None:
        xp = np.clip(x.copy(), lb, ub)
        rows = x.shape[0]
        b_inds = [i for i in range(rows) if not np.all(xp[i]==x[i])]
        if len(b_inds) == rows: # outside boundary = zero exp. imp.
            return np.zeros((rows, 1))
        e_inds = [i for i in range(x.shape[0]) \
            if np.any(np.all(np.abs(x[i] - self.X) <= 1e-9, axis=1))]
        if len(e_inds) == rows: # violates xtol = zero exp. imp.
            return np.zeros((rows, 1))
    if cfunc is not None:
        c_inds = []
        if xp.shape[0] == 1:
            if not cfunc(xp[0], *cargs, **ckwargs):
                return np.zeros((1,1))
        else:
            c_inds = [i for i in range(xp.shape[0]) if not cfunc(xp[i], *cargs, **ckwargs)]
            if len(c_inds) == rows:
                return np.zeros((rows, 1))
    y, std_dev = self.predict(xp) 
    f_best = obj_sense * np.max(obj_sense * self.Y)
    u =  obj_sense * (y - f_best) / std_dev
    sinds = [i for i in range(std_dev.shape[0]) if std_dev[i] == 0] # to get rid of NaN issues
    u[sinds] = 1e30
    # normal cumulative distribution function
    PHI = (0.5 * ERF(u/np.sqrt(2.0)))+0.5 
    # normal density function
    phi = 1/np.sqrt(2.0*np.pi)*np.exp(-u**2/2.0) 
    ei =  std_dev * ((u * PHI) + (phi)) 
    ei[sinds] = 0
    if lb is not None:
        ei[e_inds] = 0
        ei[b_inds] = 0
    if cfunc is not None:
        ei[c_inds] = 0
    return ei

def Forrester_BO_Function(x):
    
#    if (x > 1.3999):
#        return 100000
#    elif (x < 0.3001):
#        return 100000
#    elif ((x > 0.76) and (x < 0.77)):
#        return 100000
#    else:
    return (6*x - 2)**2 * np.sin(12*x - 4)

if __name__ == '__main__':
    
    current = os.getcwd()

    
    if (sampling == 'latin_forrester'):
        
        start_sim = time.time()
        print('Forrester run.' + ' in ' + current)
        subprocess.call(['rm', '-r', current + '/data/Forrester/case_local/'])
        subprocess.call(['mkdir', '-p', current + '/data/Forrester/case_local/'])
        print("main: removed old case, copied new case")
        
    else:
      
        start_sim = time.time()
        print('Demo run for Hydro case.')
        
        seed = 1005
        np.random.seed(seed)
        prob = HeadCell({})
    
        # sets values and removes olds case directory
        subprocess.call(['rm', '-r', prob.case_path])
        subprocess.call(['rm', '-r', prob.mesh_path])
        subprocess.call(['cp', '-r', prob.source_case, prob.case_path])
        print("main: removed old case, copied new case")
        subprocess.call(['mkdir', prob.mesh_path])
        
        # get upper and lower bounds:
        lb, ub = prob.get_decision_boundary() 
        print("lb", lb)
        print("ub", ub)
    
        sim_id = 20191002
        init_file_name = None
        n_samples = 10              # (11n - 1) 
        budget = 90                 # (11n - 1) + (2/3)(11n-1)
        
        layout = Ellipse(lb, ub)
     
    #initialisation()
    #no_queue()
    
    freeze_support()
    queue()
    print("Number of simultaneous runs, ", no_of_nodes)
    print("Time taken (seconds), ", (time.time()-start_sim))
    
    
    # Serial optimiser
    start = time.time()
    n_obj = 1
    n_dim = 1
    n_samples = 5 
    budget = n_samples + 3
    lb = 0.3
    ub = 1.4
    res = IscaOpt.Optimiser.EMO(Forrester_BO_Function, \
                                settings={'n_dim':n_dim, \
                                'obj_sense':[-1]*n_obj, \
                                'n_obj':n_obj, \
                                'method_name':'EGO', \
                                'lb':np.array([lb]), \
                                'ub':np.array([ub]), \
                                'n_samples':n_samples, \
                                'budget':budget, \
                                'visualise':True, \
                                'multisurrogate':False, \
                                'init_file':current + '/initial_samples.npz'})
    
    print('Objective function value:', res)
    print('Optimiser Time taken:', time.time()-start, ' seconds.')
    