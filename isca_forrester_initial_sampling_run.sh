#!/bin/sh

module load Anaconda3/4.3.0
source activate py3

python -u forrester_initialisation_parallel.py > log.python 2>&1 &
echo "Python running in background with nohup"

