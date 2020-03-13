#!/bin/bash --login

# Name of job
#PBS -N pw_script_hydro                                                                                                                                      
# Select wall time 10 hours                                                                                            
#PBS -l walltime=00:09:59                                                                                                                                                                                                                  
# Use the Pascal node                                                                                                  
#PBS -q pascalq                                                                                                                        
# Output and error files
#PBS -o 10.141.0.1:/lustre/home/ex-aroberts/Pointwise_User/XX2_pointwise_original/log.pointwise
#PBS -e 10.141.0.1:/lustre/home/ex-aroberts/Pointwise_User/XX2_pointwise_original/err.pointwise

# Change to current working directory
source $HOME/pointwise_directory.sh
cd $POINTWISE_DIRECTORY

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------
                                                                                                                                       
$HOME/Pointwise/PointwiseV18.0R2/pointwise -b Hydro_V18_3_tray_APR_grit_pot_parameterised_ellipse_correction.glf

