#!/bin/sh

# Run this script like this: 
# In HeadCell_Sampling.py: environment = 'isca'
# In run_case.py: platform = 'isca'
# $ sh isca_initial_sampling_run.sh > log.shell_script 2>&1 &

# Load Python 
module load Anaconda3/4.3.0
source activate py3
module load OpenFOAM/v1812-foss-2018b
source /gpfs/ts0/home/apr207/OpenFOAM/OpenFOAM-v1812/etc/bashrc

python -u HeadCell_Sampling.py > log.python 2>&1 &
echo "Python running in background with nohup"

