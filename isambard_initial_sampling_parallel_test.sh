#!/bin/bash

#PBS -N job_headcell

# Select 1 nodes (maximum of 64 cores)
#PBS -l select=1:ncpus=8

# Select wall time to 40 minutes
#PBS -l walltime=00:39:59

# Use the arm nodes
#PBS -q arm

# Load Python 
module load cray-python/3.6.5.6
source ~/myPy/bin/activate

# Load modules for currently recommended OpenFOAM 1812 build
module swap PrgEnv-cray PrgEnv-allinea
module load PrgEnv-allinea

# Change to directory that script was submitted from
export PBS_O_WORKDIR=$(readlink -f $PBS_O_WORKDIR)
export OMP_NUM_THREADS=1
cd $PBS_O_WORKDIR

# Load the environment variables for OpenFOAM v1812 build
export OPENFOAM_DIR=/home/ex-aroberts/OpenFOAM/OpenFOAM-v1812
export PATH=$PATH:$OPENFOAM_DIR/bin/
source $OPENFOAM_DIR/etc/bashrc

python -u run_case.py > log.python 2>&1

