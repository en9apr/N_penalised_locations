#!/bin/sh

# Run this script like this: 
# In HeadCell_Sampling.py: environment = 'isambard'
# In run_case.py: platform = 'isambard'
# $ sh isambard_initial_sampling_run.sh > log.shell_script 2>&1 &

# Load Python 
module load cray-python/3.6.5.6
source ~/myPy/bin/activate

# Load modules for currently recommended OpenFOAM 1812 build
module swap PrgEnv-cray PrgEnv-allinea
module load PrgEnv-allinea

# Load the environment variables for OpenFOAM v1812 build
export OPENFOAM_DIR=/home/ex-aroberts/OpenFOAM/OpenFOAM-v1812
export PATH=$PATH:$OPENFOAM_DIR/bin/
source $OPENFOAM_DIR/etc/bashrc

python -u HeadCell_Sampling.py > log.python 2>&1 &
echo "Python running in background with nohup"

