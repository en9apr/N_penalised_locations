#!/bin/sh

# This script is run automatically

#PBS -V
#PBS -d .
#PBS -q ptq
#PBS -l walltime=00:39:59
#PBS -A Research_Project-T110712
#PBS -l nodes=1:ppn=8
#PBS -m e -M A.P.Roberts@exeter.ac.uk

module load Anaconda3/4.3.0
source activate py3
module load OpenFOAM/v1812-foss-2018b
source /gpfs/ts0/home/apr207/OpenFOAM/OpenFOAM-v1812/etc/bashrc

cd $PBS_O_WORKDIR
. $WM_PROJECT_DIR/bin/tools/RunFunctions

python -u run_case.py > log.python 2>&1

