#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:55:13 2019

@author: andrew
"""
import subprocess
import time
import os
from os.path import basename
import shutil
import re
from PyFoam.Execution.UtilityRunner import UtilityRunner
import math as mt
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile
from pathlib import Path
import fasteners
import random

#platform = 'isambard'
#platform = 'isambard_test'
#platform = 'isca'
#platform = 'isca_test'
#platform = 'isca_forrester'
platform = None

sampling = 'latin_forrester'

# follow a log file:
def follow(thefile):
    thefile.seek(0,2)
    while True:
        line = thefile.readline()
        if not line:
            time.sleep(0.1)
            continue       
        yield line
        
# compute the average of a list:        
def Average(lst): 
    return sum(lst) / len(lst)



# potentialFoam, simpleFoam, parcelFoam:
def OpenFOAM():
    
    current = os.getcwd()    
    utility1="decomposePar"
    utility2="renumberMesh"
    utility3="transformPoints"
    utility4="reconstructPar"
    solver1="icoUncoupledKinematicCustomInteractionFoam"
    solver4="pressureDropSimpleFoam"                        
    solver5="potentialFoam" 
    
    # platform selection:
    if platform == 'blades':
        total_iterations = 8000
        averaging_iterations = 1000
        ncores=28
        subprocess.call(['cp', current + '/system/decomposeParDict_blades', current + '/system/decomposeParDict'])
        print("OpenFOAM(): copied decomposeParDict_blades to decomposeParDict") 
    
    elif platform == 'local':
        total_iterations = 8000
        averaging_iterations = 1000
        ncores=8
        subprocess.call(['cp', current + '/system/decomposeParDict_local', current + '/system/decomposeParDict'])
        print("OpenFOAM(): copied decomposeParDict_local to decomposeParDict") 
        
    elif platform == 'isca':
        total_iterations = 8000
        averaging_iterations = 1000
        ncores=16
        subprocess.call(['cp', current + '/system/decomposeParDict_isca', current + '/system/decomposeParDict'])
        print("OpenFOAM(): copied decomposeParDict_isca to decomposeParDict") 
        
    elif platform == 'isambard':
        total_iterations = 8000
        averaging_iterations = 1000 # ignoring 1st 999 iterations
        ncores=64
        subprocess.call(['cp', current + '/system/decomposeParDict_isambard', current + '/system/decomposeParDict'])
        print("OpenFOAM(): copied decomposeParDict_isambard to decomposeParDict")

    elif platform == 'isambard_test':
        total_iterations = 10
        averaging_iterations = 2 # ignoring 1st iteration
        ncores=8
        subprocess.call(['cp', current + '/system/decomposeParDict_isambard_test', current + '/system/decomposeParDict'])
        print("OpenFOAM(): copied decomposeParDict_isambard_test to decomposeParDict")
        
    elif platform == 'isca_test':
        total_iterations = 10
        averaging_iterations = 2 # ignoring 1st iteration
        ncores=8
        subprocess.call(['cp', current + '/system/decomposeParDict_isca_test', current + '/system/decomposeParDict'])
        print("OpenFOAM(): copied decomposeParDict_isca_test to decomposeParDict")
        
    else:
        # local is default
        total_iterations = 8000
        averaging_iterations = 1000
        ncores=8
        subprocess.call(['cp', current + '/system/decomposeParDict_local', current + '/system/decomposeParDict'])
        print("OpenFOAM(): copied decomposeParDict_local to decomposeParDict") 


    # dir_name is the folder name for the current directory
    dir_name = basename(current)

    # Copy the mesh into the current directory:
    shutil.rmtree(current + '/constant/polyMesh')
    subprocess.call(['cp', '-r', '../../meshes/' + dir_name + '/constant/polyMesh', current + '/constant/polyMesh'])
    print("OpenFOAM(x): pointwise mesh copied from subdirectory")


    # Remove the mesh from the directory to save disk space:
    subprocess.call(['rm', '-r', '../../meshes/' + dir_name + '/'])
    print("OpenFOAM(): pointwise mesh removed from subdirectory")

    # reset initial conditions
    subprocess.call(['rm', '-r', current + '/0'])
    subprocess.call(['cp', '-r', current + '/0_orig', current + '/0'])
    print("OpenFOAM(): removed 0 folder, copied 0_orig to 0")
    
    # clear case:
    subprocess.call(['pyFoamClearCase.py', current, '--processors-remove']) 
    print("OpenFOAM(): pyFoamClearCase completed")
        
    # new case:
    print("#### START OF NEW CASE ####")
    print("OpenFOAM(): case " + dir_name)
    
    # Correct dictionaries for angle of underflow velocity:
    velocity_magnitude = 1.3847581294
    
    f = open('decision_vector.txt')
    lines = f.readlines()
    angle = float(lines[2])
    f.close()
    
    print("OpenFOAM(): The underflow angle is", angle)
    
    x_velocity = velocity_magnitude*mt.cos(angle*(mt.pi/180.0))
    
    z_velocity = velocity_magnitude*mt.sin(angle*(mt.pi/180.0))
         
    f=ParsedParameterFile("0/U")
    
    velocity_vector = 'uniform (' + str(x_velocity) + ' ' + '0.0' + ' ' + str(z_velocity) + ')'
     
    for b in f["boundaryField"]:
        if "outflow_plate" in b:
            f["boundaryField"][b]["value"]=velocity_vector
     
    f.writeFile()
    print("OpenFOAM(): corrected underflow velocity")
    
    #transformPoints
    subprocess.call([utility3, '-scale', " (0.0254 0.0254 0.0254) ", "-case", current], cwd=current, \
                  stdout = open(current + '/log.transformPoints', 'w'), \
                  stderr = open(current + '/err.transformPoints', 'w'))                     
    print("OpenFOAM(): transformPoints completed")    
    
    if((platform == 'isambard_test') or (platform == 'isca_test')):
    
        # copy simpleFoam control dictionary:
        subprocess.call(['cp', '-r', current + '/system/controlDict_simple_test', current + '/system/controlDict'])
        print("OpenFOAM(): copied controlDict_simple_test")
    
    else:
        # copy simpleFoam control dictionary:
        subprocess.call(['cp', '-r', current + '/system/controlDict_simple', current + '/system/controlDict'])
        print("OpenFOAM(): copied controlDict_simple")
        
    # renumberMesh:
    subprocess.call([utility2, '-case', current, '-overwrite'], cwd=current, \
                  stdout = open(current + '/log.renumberMesh', 'w'), \
                  stderr = open(current + '/err.renumberMesh', 'w'))
    print("OpenFOAM(): renumberMesh completed")

    # decompose the domain:
    subprocess.call([utility1, '-case', current], cwd=current, \
                  stdout = open(current + '/log.decomposePar', 'w'), \
                  stderr = open(current + '/err.decomposePar', 'w'))
    print("OpenFOAM(): decomposePar completed")

    if(platform == 'isambard') or (platform == 'isambard_test'):

        # run potentialFoam:
        subprocess.call(['aprun', '-n', str(ncores), solver5, '-initialiseUBCs', '-parallel','-case', current], cwd=current, \
                          stdout = open(current + '/log.potentialFoam', 'w'), \
                          stderr = open(current + '/err.potentialFoam', 'w'))
        print("OpenFOAM(): potentialFoam completed")
    
        # run simpleFoam in the background using Popen:
        subprocess.Popen(['aprun', '-n', str(ncores), solver4, '-parallel','-case', current], cwd=current, \
                          stdout = open(current + '/log.pressureDropSimpleFoam', 'w'), \
                          stderr = open(current + '/err.pressureDropSimpleFoam', 'w'))

    else:
        
        # run potentialFoam:
        subprocess.call(['mpirun', '-np', str(ncores), solver5, '-parallel','-case', current], cwd=current, \
                          stdout = open(current + '/log.potentialFoam', 'w'), \
                          stderr = open(current + '/err.potentialFoam', 'w'))
        print("OpenFOAM(): potentialFoam completed")
    
        # run simpleFoam in the background using Popen:
        subprocess.Popen(['mpirun', '-np', str(ncores), solver4, '-parallel','-case', current], cwd=current, \
                          stdout = open(current + '/log.pressureDropSimpleFoam', 'w'), \
                          stderr = open(current + '/err.pressureDropSimpleFoam', 'w'))


    # open log file:    
    logfile = open(current + '/log.pressureDropSimpleFoam', 'r')
    loglines = follow(logfile)
    
    # regular expressions:
    linearRegExp="^(.+):  Solving for p, Initial residual = (.+), Final residual = (.+), No Iterations (.+)$"
    linearRegExp2="^time step continuity errors : sum local = (.+), global = (.+), cumulative = (.+)$"
    linearRegExp4="^Pressure_coefficient_inlet_underflow = (.+)$"    
    linearRegExp3="^Pressure_coefficient_inlet_overflow = (.+)$"

    
    # data structures:
    k=1
    i=0
    nNonOrthogonalCorrectors = 3
    z = nNonOrthogonalCorrectors+1
    
    residuals = []
    continuity = []
    pressure_underflow = []
    pressure_overflow = []
    
    
    residuals_mean = []
    continuity_mean = []
    pressure_underflow_mean = []
    pressure_overflow_mean = []
    
    
    percent_continuity = []
    percent_residuals = []
    percent_underflow_pressure = []
    percent_overflow_pressure = []
    
    
    # read log file:
    for line in loglines:
        m=re.compile(linearRegExp).match(line.strip())
        m2=re.compile(linearRegExp2).match(line.strip())
        m4=re.compile(linearRegExp4).match(line.strip())
        m3=re.compile(linearRegExp3).match(line.strip())
        
        
        if m!=None:
            if (z % 4 == 0):
                residuals.append(float(m.group(2)))
                if ((platform == 'isambard_test') or (platform == 'isca_test')):
                    print("residuals", float(m.group(2)))
            z+=1
            
        if m2!=None:
            continuity.append(float(m2.group(3)))
            if ((platform == 'isambard_test') or (platform == 'isca_test')):
                print("continuty", float(m2.group(3)))
            
        if m4!=None:
            pressure_underflow.append(float(m4.group(1)))
            if ((platform == 'isambard_test') or (platform == 'isca_test')):
                print("underflow", float(m4.group(1)))
            
        if m3!=None:
            pressure_overflow.append(float(m3.group(1)))
            if ((platform == 'isambard_test') or (platform == 'isca_test')):
                print("overflow", float(m3.group(1)))
            k+=1

            
        if (k > total_iterations):
            print("OpenFOAM(): pressureDropSimpleFoam completed it's iterations, where len(residuals) = ", len(residuals), \
                  "len(continuity) = ", len(continuity),"len(overflow) = ", len(pressure_overflow),"len(underflow) = ", len(pressure_underflow))
            with open(current + "/convergence.txt", "a") as myfile_convergence:
                myfile_convergence.write((str(0)) + '\n')
            print("OpenFOAM(): written convergence failure to a file") 
            break
            
        if (m3!=None) and ((k % averaging_iterations == 0) and (k > averaging_iterations)):
            
            # ignore the first 1000 iterations
            residuals_values = residuals[averaging_iterations:]
            continuity_values = continuity[averaging_iterations:]
            pressure_underflow_values = pressure_underflow[averaging_iterations:]
            pressure_overflow_values = pressure_overflow[averaging_iterations:]
            
            
            residuals_mean.append(Average(residuals_values))
            continuity_mean.append(Average(continuity_values))
            pressure_underflow_mean.append(Average(pressure_underflow_values))
            pressure_overflow_mean.append(Average(pressure_overflow_values))
            
            i+=1
            
            if ((platform == 'isambard_test') or (platform == 'isca_test')):
                print("mean residual", Average(residuals_values))
                print("mean continuity", Average(continuity_mean))
                print("mean underflow", Average(pressure_underflow_mean))
                print("mean overflow", Average(pressure_overflow_mean))
                print("k", k)
                print("i", i)
            
            if (i > 1):   
                percent_residuals.append(abs((residuals_mean[i-1]-residuals_mean[i-2]) / residuals_mean[i-2]))
                percent_continuity.append(abs((continuity_mean[i-1]-continuity_mean[i-2]) / continuity_mean[i-2])) 
                percent_underflow_pressure.append(abs((pressure_underflow_mean[i-1]-pressure_underflow_mean[i-2]) / pressure_underflow_mean[i-2]))
                percent_overflow_pressure.append(abs((pressure_overflow_mean[i-1]-pressure_overflow_mean[i-2]) / pressure_overflow_mean[i-2]))
                
                
                # the difference is less than 10 percent, break:
                if (abs(percent_residuals[-1]) < 0.1) and ((abs(percent_continuity[-1]) < 0.1) and ((abs(percent_overflow_pressure[-1]) < 0.1) and (abs(percent_underflow_pressure[-1]) < 0.1))):   
                   subprocess.call(['pyFoamWriteDictionary.py', current + '/system/controlDict', '"stopAt"', 'writeNow'])
                   print("OpenFOAM(): pressureDropSimpleFoam converged")
                   with open(current + "/convergence.txt", "a") as myfile_convergence:
                       myfile_convergence.write((str(1)) + '\n')
                   print("OpenFOAM(): written convergence success to a file")    
                   break
               
    # sleep for 5 min in order to write the data to disk:
    time.sleep(300)
    print('OpenFOAM(): 5min pause finished')

    # write means to a file
    with open(current + "/means.txt", "a") as myfile_means:
        myfile_means.write('iterations'+','+'residuals_mean'+','+'continuity_mean'+','+'pressure_overflow_mean'+','+'pressure_underflow_mean'+'\n')
        for j in range(0,len(residuals_mean)):
            myfile_means.write(str(j*averaging_iterations+averaging_iterations)+','+str(residuals_mean[j])+','+str(continuity_mean[j])+','+str(pressure_overflow_mean[j])+','+str(pressure_underflow_mean[j])+'\n')      
            myfile_means.write(str((j+1)*averaging_iterations+averaging_iterations)+','+str(residuals_mean[j])+','+str(continuity_mean[j])+','+str(pressure_overflow_mean[j])+','+str(pressure_underflow_mean[j])+'\n')
            
    # write values to a file
    with open(current + "/values.txt", "a") as myfile_values:
        myfile_values.write('residuals'+','+'continuity'+','+'pressure_overflow'+','+'pressure_underflow'+'\n')
        for m in range(0,len(residuals)):
            myfile_values.write(str(residuals[m])+','+str(continuity[m])+','+str(pressure_overflow[m])+','+str(pressure_underflow[m])+'\n') 
    print("OpenFOAM(): written means.txt and values.txt")

    # run reconstructPar:
    subprocess.call([utility4, '-latestTime', "-case", current], cwd=current, \
                      stdout = open(current + '/log.reconstructPar', 'w'), \
                      stderr = open(current + '/err.reconstructPar', 'w'))
    print("OpenFOAM(): reconstructPar completed")

    # copy the last to the first:
    subprocess.call(['pyFoamCopyLastToFirst.py', current, current])
    print("OpenFOAM(): copied latestTime to 0")
    
    # clear the case:
    subprocess.call(['pyFoamClearCase.py', current, '--processors-remove'])
    print("OpenFOAM(): cleared parent directory")
    
    
    if((platform == 'isambard_test') or (platform == 'isca_test')):
    
        # copy control dictionary to current directory:
        subprocess.call(['cp', current + '/system/controlDict_parcel_test', current + '/system/controlDict'])
        print("OpenFOAM(): copied controlDict_parcel_test to controlDict")
        
    else:
        # copy control dictionary to current directory:
        subprocess.call(['cp', current + '/system/controlDict_parcel', current + '/system/controlDict'])
        print("OpenFOAM(): copied controlDict_parcel to controlDict")
        
    # decompose the domain:
    subprocess.call([utility1, '-case', current], cwd=current, \
                  stdout = open(current + '/log.decomposeParParcel', 'w'), \
                  stderr = open(current + '/err.decomposeParParcel', 'w'))
    print("OpenFOAM(): decomposeParParcel completed")
    
    
    if((platform == 'isambard') or (platform == 'isambard_test')):
        
        # run icoUncoupledKinematicCustomInteractionFoam:
        subprocess.call(['aprun', '-n', str(ncores), solver1, '-parallel', '-case', current], cwd=current, \
                          stdout = open(current + '/log.icoUncoupledKinematicCustomInteractionFoam', 'w'), \
                          stderr = open(current + '/err.icoUncoupledKinematicCustomInteractionFoam', 'w'))
        print("OpenFOAM(): icoUncoupledKinematicCustomInteractionFoam completed")
        
    else:
        # run icoUncoupledKinematicCustomInteractionFoam:
        subprocess.call(['mpirun', '-np', str(ncores), solver1, '-parallel', '-case', current], cwd=current, \
                          stdout = open(current + '/log.icoUncoupledKinematicCustomInteractionFoam', 'w'), \
                          stderr = open(current + '/err.icoUncoupledKinematicCustomInteractionFoam', 'w'))
        print("OpenFOAM(): icoUncoupledKinematicCustomInteractionFoam completed")        
        
    # run reconstructPar:
    subprocess.call([utility4, '-latestTime', "-case", current], cwd=current, \
                      stdout = open(current + '/log.reconstructParParcel', 'w'), \
                      stderr = open(current + '/err.reconstructParParcel', 'w'))
    print("OpenFOAM(): reconstructParParcel completed")
       
    # run reconstructPar:
    subprocess.call([utility4, '-latestTime', '-lagrangianFields', \
        '(collisionRecordsPairAccessed collisionRecordsPairData collisionRecordsPairOrigIdOfOther collisionRecordsPairOrigProcOfOther collisionRecordsWallAccessed collisionRecordsWallData collisionRecordsWallPRel)', \
        "-case", current], cwd=current, \
        stdout = open(current + '/log.reconstructParLagrangian', 'w'), \
        stderr = open(current + '/err.reconstructParLagrangian', 'w'))
    print("OpenFOAM(): reconstructParLagrangian completed")
    
    if((platform == 'isambard_test') or (platform == 'isca_test')):    
    
        # copy the control dictionary to the current directory:
        subprocess.call(['cp', current + '/system/controlDict_postprocess_test', current + '/system/controlDict'])
        print("OpenFOAM(): copied controlDict_postprocess to controlDict")
    
    else:    

        # copy the control dictionary to the current directory:
        subprocess.call(['cp', current + '/system/controlDict_postprocess', current + '/system/controlDict'])
        print("OpenFOAM(): copied controlDict_postprocess to controlDict")
        
    with open(current + "/pressure_overflow.txt", "a") as myfile_pressure_overflow:
        myfile_pressure_overflow.write(str(float(Average(pressure_overflow[-averaging_iterations:])))+ '\n')
    print("OpenFOAM(): written pressure_overflow to a file")


# one timestep of parcelfoam:    
def RunUtilities():
    
    current = os.getcwd()
    
    solver1 = "icoUncoupledKinematicCustomInteractionFoam"
    
    # run parcelFoam for one timestep:
    mCalculated = UtilityRunner(argv=[solver1, "-case", current], silent=True, logname="ParticleEscape")
    print("RunUtilities(x): icoUncoupledKinematicCustomInteractionFoam run for 1 timestep")
    
    mCalculated.add("massEscape", "- escape                      = outflow_top = (%f%)", idNr=1)
    mCalculated.add("massIntroduced", "- mass introduced             = (%f%)", idNr=1)
    mCalculated.start()
    massEscape1=UtilityRunner.get(mCalculated, "massEscape")[0]
    print("RunUtilities(): read massEscape to a number", str(massEscape1))
    
    massIntro1=UtilityRunner.get(mCalculated, "massIntroduced")[0]
    print("RunUtilities(): read massIntroduced to a number", str(massIntro1))
    
    if((platform == 'isambard_test') or (platform == 'isca_test')): 
        
        subprocess.call(['rm', '-rf', current + '/0.011'])
        print("RunUtilities(): removed 0.011 directory")
        
        # remove redundant files    
        subprocess.call(['pyFoamClearCase.py', current, '--after=0.01', '--processors-remove'])
        subprocess.call(['rm', '-rf', current + '/0'])
        print("RunUtilities(): removed processor directories and 0 directory")
        
    else: 
        
        subprocess.call(['rm', '-rf', current + '/262.001'])
        print("RunUtilities(): removed 262.001 directory") 
        
        # remove redundant files
        subprocess.call(['pyFoamClearCase.py', current, '--after=262', '--processors-remove'])
        subprocess.call(['rm', '-rf', current + '/0'])
        print("RunUtilities(): removed processor directories and 0 directory")
        
    with open(current + "/efficiency.txt", "a") as myfile_efficiency:
        myfile_efficiency.write(str((float(massIntro1)-float(massEscape1))/float(massIntro1))+ '\n')
    
    print("RunUtilities(): written efficiency to a file")
    
    
   # test function
def Forrester():
    
    current = os.getcwd()
    
    x = np.loadtxt(current + "/decision_vector.txt")
    print("x=", x)
    objective_function = (6*x - 2)**2 * np.sin(12*x - 4)
    print("objective function", objective_function)
    #time.sleep(5)
    #time.sleep(random.randint(5, 10))
    with open(current + "/objective_function.txt", "a") as objective_function_file:
        objective_function_file.write(str((float(objective_function)))+ '\n')
#    return (6*x - 2)**2 * np.sin(12*x - 4)

 
# define the cost function: this can take 11hours 10minutes on ISCA (say 12 hours) -  27 cases: 13.5 days (I gave it 2 weeks)
def hydro_1D():
    """
    A mock test function that accepts a 1D decision variable and updates the 
    given layout and returns a function value.
    """
    print("hydro_1D(): evaluating cost function")
    
    if(sampling == 'latin_forrester'):
        
        Forrester()
    
    else:
    
        # most of CFD run (potential, simple, parcel)
        OpenFOAM()
        # expensive cost function (one timestep of parcel): 
        RunUtilities()


if __name__=="__main__":
    
    import numpy as np

    # write the files neccessary for the 2 objectives   
    hydro_1D()

    # get the current directory
    current = os.getcwd()
    print("main(): current directory:", current)

    if(sampling == 'latin_forrester'):
        decision_vector = np.loadtxt(current + "/decision_vector.txt")
        objective_function = np.loadtxt(current + "/objective_function.txt")

        # the path to the sim file:
        p = Path(current + "/decision_vector.txt").parents[4]
        print("main(): path to initial samples file:", p)
            
        # load the .npz file:
        sim_file = "initial_samples.npz"
        a_lock = fasteners.InterProcessLock(str(p) + '/' + sim_file)
        
        # load the data and write the new data
        while True:
            unlocked = a_lock.acquire(blocking=False)
            try:
                if unlocked:
                    print('I have the lock')
                    #time.sleep(5)
                    
                    data = np.load(str(p) + '/' + sim_file)
                    print('data file loaded')
                    X = data['arr_0']
                    Y = data['arr_1']
                    #X = np.array([X])
                    #Y = np.array([Y])
                    print("X", X)
                    print("Y", Y)
                    X_current = np.array([decision_vector])
                    Y_current = np.array([objective_function])
                    print("X_current", X_current)
                    print("Y_current", Y_current)
                    
                    hpv_initial = []
                    time_initial = 0.0
                    print("len(X)", len(X))
                    print("len(Y)", len(Y))
                    
                    if len(X) > 0:
                        print("X array is not empty")
                        X_new = np.vstack((X, X_current))
                        print("len(x)= ", len(X))
                        print('appended ', X_current, ' to ', X)
                        print('this gives ', X_new)
                    else:
                        print("X array is empty")
                        print("len(x)= ", len(X))
                        X_new = X_current
                        print('set ', X_new, ' to ', X_current)
                    
                    if len(Y) > 0:
                        print("Y array is not empty")
                        Y_new = np.vstack((Y, Y_current))
                        print("len(y)= ", len(Y))
                        print('appended ', Y_current, ' to ', Y)
                        print('this gives ', Y_new)
                    else:
                        print("Y array is empty")
                        print("len(y)= ", len(Y))
                        Y_new = Y_current
                        print('set ', Y_new, ' to ', Y_current)
                        
                        
                    try:     
                        np.savez(str(p) + '/' + sim_file, X_new, Y_new, hpv_initial, time_initial)
                        print('Data saved in file: ', sim_file, ' directory ', str(p), ' X ', X_new, ' Y ', Y_new)
                    except Exception as e:
                        print(e)
                        print('Data saving failed.')
                        
                    # trying to ensure data writing succeeds:
                    time.sleep(5)
                    
                else:
                    print('I do not have the lock')
                    time.sleep(5)
            finally:
                if unlocked:
                    a_lock.release()
                    break                


    else:

        # load the decision vector and 2 objectives:
        decision_vector = np.loadtxt('decision_vector.txt')
        efficiency = np.loadtxt('efficiency.txt')
        pressure_overflow = np.loadtxt('pressure_overflow.txt')
        convergence = np.loadtxt('convergence.txt')
        
        # the path to the sim file:
        p = Path(current + "/decision_vector.txt").parents[4]
        print("main(): path to initial samples file:", p)
            
        # load the .npz file:
        sim_file = "initial_samples.npz"
        a_lock = fasteners.InterProcessLock(str(p) + '/' + sim_file)
        
        # load the data and write the new data
        while True:
            unlocked = a_lock.acquire(blocking=False)
            try:
                if unlocked:
                    print('I have the lock')
                    time.sleep(20)
                    
                    data = np.load(str(p) + '/' + sim_file)
                    
                    X = data['arr_0']
                    Y = data['arr_1']
                    C = data['arr_4']
                    
                    X_current = decision_vector
                    Y_current = np.array([efficiency, pressure_overflow])
                    C_current = np.array([convergence])
                    
                    hpv_initial = []
                    time_initial = 0.0
                    
                    if len(X) > 0:
                        X_new = np.vstack((X, X_current))
                        #print('appended ', X_current, ' to ', X)
                        #print('this gives ', X_new)
                    else:
                        X_new = X_current
                        #print('set ', X_new, ' to ', X_current)
                    
                    if len(Y) > 0:
                        Y_new = np.vstack((Y, Y_current))
                        #print('appended ', Y_current, ' to ', Y)
                        #print('this gives ', Y_new)
                    else:
                        Y_new = Y_current
                        #print('set ', Y_new, ' to ', Y_current)
                        
                    if len(C) > 0:
                        C_new = np.vstack((C, C_current))
                        #print('appended ', Y_current, ' to ', Y)
                        #print('this gives ', Y_new)
                    else:
                        C_new = C_current
                        #print('set ', Y_new, ' to ', Y_current)
                        
                    try:     
                        np.savez(str(p) + '/' + sim_file, X_new, Y_new, hpv_initial, time_initial, C_new)
                        print('Data saved in file: ', sim_file, ' directory ', str(p), ' X ', X_new, ' Y ', Y_new, ' C ', C_new)
                    except Exception as e:
                        print(e)
                        print('Data saving failed.')
                    
                else:
                    print('I do not have the lock')
                    time.sleep(20)
            finally:
                if unlocked:
                    a_lock.release()
                    break                
                