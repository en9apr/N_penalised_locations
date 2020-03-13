#! /usr/bin/env python

# imports
import shutil
import os
try:
    from .InitialTEST import InitialTEST#, isCallback
    from .support import RemoveCase, RestoreCase, nostdout, suppress_stdout
except Exception as e:
    print(e)
    from InitialTEST import InitialTEST#, isCallback
    from support import RemoveCase, RestoreCase, nostdout
from PyFoam.Execution.ConvergenceRunner import ConvergenceRunner
from PyFoam.Execution.UtilityRunner import UtilityRunner
from PyFoam.LogAnalysis.BoundingLogAnalyzer import BoundingLogAnalyzer
from PyFoam.Applications.CloneCase import CloneCase
from PyFoam.Applications.Runner import Runner
from PyFoam.Execution.BasicRunner import BasicRunner
from PyFoam.Applications.CopyLastToFirst import CopyLastToFirst
from PyFoam.Applications.ClearCase import ClearCase
from PyFoam.Execution.ParallelExecution import LAMMachine
import subprocess
import numpy as np
from stl import mesh
from stl import stl
import fnmatch
import csv
import os, sys
from contextlib import redirect_stdout
import sys, os, io, pdb
import warnings

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class BasicHeatExchangerRun(InitialTEST):

    """
    CFD test problem
    1) Construct the mesh
    2) Run checkMesh on latest mesh
    3) Run steady state case, (no optimisation)
    """

    # class attributes
    # these attributes are likely to be the same accross all instances.
    #Solvers
    solver1="heatedFoam"
    solver2="snappyHexMesh"
    solver3="createPatch"
    solver4="mapFields"
    solver5="surfaceFeatureExtract"
    solver6="extrudeMesh"
    #utilities
    checkingmesh="checkMesh"
    solver7="simpleFoam"
    #CostFunction postprocessing tools
    pCmd="calcPressureDifference_heatexchanger"
    tCmd="calcTemperatureDifference"


    def init(self):
        self.setParameters(solver="checkMesh",
                           sizeClass=self.size_class,
                           minimumRunTime=self.min_run_time,
                           casePath=self.case_path)

    def prepare_case(self, source_case, verbose=False):
        if verbose:
            self.__prepare_case(source_case)
        else:
            with nostdout():
                self.__prepare_case(source_case)

    def __prepare_case(self, source_case):
        # remove any previous case directory
        RemoveCase(self.case_path)
        # restore case from source before running for the first time
        RestoreCase(source_case, self.case_path)
        # Run CFD on base case
        self.run()


    def postRunTestCheckConverged(self):
        '''
        self.isNotEqual(
            value=self.runInfo()["time"],
            target=self.controlDict()["endTime"],
            message="Reached endTime -> not converged")
        '''
        self.shell("cp -r 0 0_orig") #initial fields
        #self.shell("pyFoamCopyLastToFirst.py . .")
        #self.shell("pyFoamClearCase.py .")
        CloneCase(args=(self.case_path, self.case_path+"heat_exchange")) #Only works with Python3
        self.shell("cp -r 0_orig heat_exchange/") #initial fields
        self.shell("cp -r heat_exchange/constant/polyMesh heat_exchange/constant/polyMesh_backup")

    def SnappyHexMeshrun(self):
        subprocess.call(['rm', '-r', self.case_path+'constant/polyMesh'])
        subprocess.call(['cp', '-r', self.case_path+'heat_exchange/constant/polyMesh_backup', self.case_path+'/constant/polyMesh'])
        subprocess.call(['rm', '-r', self.case_path+'0'])
        subprocess.call(['cp', '-r', self.case_path+'heat_exchange/0_orig', self.case_path+'0'])
        surface = BasicRunner(argv=[self.solver5,"-case", self.case_path], silent=False)
        surface.start()
        snappy = BasicRunner(argv=[self.solver2,"-overwrite","-case",self.case_path], silent=False)
        snappy.start()
        extrude = BasicRunner(argv=[self.solver6,"-case",self.case_path], silent=False)
        extrude.start()
        check = BasicRunner(argv=[self.checkingmesh, "-latestTime","-case", self.case_path], silent=False)
        check.start()
        #merge = BasicRunner(argv=[self.solver3, "-overwrite","-case", self.case_path], silent=True) # merge STL with lowerboundary
        #merge.start()

    def Optimisationrun(self): #run simpleFoam
        run=ConvergenceRunner(BoundingLogAnalyzer(),argv=[self.solver1,"-case",self.case_path],silent=True)
        run.start()

        subprocess.call(['pyFoamCopyLastToFirst.py',self.case_path, self.case_path])
        subprocess.call(['pyFoamClearCase.py', self.case_path])
        subprocess.call(['rm', self.case_path+'0/cellLevel'])

    def RunUtilities(self, sense='single'):
        # Get the pressure difference (Using an external utility)
        pUtil=UtilityRunner(argv=[self.pCmd,"-case",self.case_path,"-latestTime"],silent=True,logname="Pressure")
        pUtil.add("PressureDifference","Pressure drop = (%f%) between inlet and outlet",idNr=1)
        pUtil.start()
        deltaP=UtilityRunner.get(pUtil,"PressureDifference")[0]

        tUtil=UtilityRunner(argv=[self.tCmd,"-case",self.case_path,"-latestTime"],silent=True,logname="Temperature")
        tUtil.add("TemperatureDifference","Temperature drop = (%f%) between inlet and outlet",idNr=1)
        tUtil.start()
        deltaT=UtilityRunner.get(tUtil,"TemperatureDifference")[0]

        if sense=="multi":

            return float(deltaT), float(deltaP)
        else:
            return float(deltaT)

    def __cost_function(self, sense='single'):
        """
        A method to run CFD for the new shape.

        Kwargs:
            sense (str): whther to return single or multi-objective values.
        """

        self.SnappyHexMeshrun()
        self.Optimisationrun()
        if sense=="single":
            t = self.RunUtilities()
            return p
        elif sense=="multi":
            t, p = self.RunUtilities(sense=sense)
            return t, p
        else:
            print("Invalid input for sense: ", sense)
            print("Available options are: 'single' or 'multi'")
            return None

    def cost_function(self, sense='single', verbose=False):
        """
        A method to run CFD for the new shape.

        Kwargs:
            sense (str): whther to return single or multi-objective values.
        """
        if verbose:
            return self.__cost_function(sense=sense)
        else:
            with nostdout():
                return self.__cost_function(sense=sense)
