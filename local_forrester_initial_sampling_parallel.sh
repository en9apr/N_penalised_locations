#!/bin/bash

# Python already loaded, just load OpenFOAM
#source $HOME/OpenFOAM/OpenFOAM-v1812/etc/bashrc FOAMY_HEX_MESH=yes export

python -u run_case.py > log.python 2>&1
echo "Python running in background with nohup"

