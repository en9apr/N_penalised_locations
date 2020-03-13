#!/bin/sh

# This script is run automatically

#PBS -V
#PBS -d .
#PBS -q ptq
#PBS -l walltime=00:05:00
#PBS -A Research_Project-T110712
#PBS -l nodes=1:ppn=1
#PBS -m e -M A.P.Roberts@exeter.ac.uk

module load Anaconda3/4.3.0
source activate py3

python -u run_case.py > log.python 2>&1
echo "Python running in background with nohup"

