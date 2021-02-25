#!/bin/bash

#PBS -l ncpus=8
#PBS -l mem=190GB
#PBS -l jobfs=200GB
#PBS -q normal
#PBS -l walltime=02:00:00
#PBS -l wd

module load use.own
module load intel-mkl/2019.3.199
module load python3/3.7.4
module load sklearn/0.24.1
module load pytorch/1.5.1

python3 train_nf.py  $PBS_NCPUS > $PBS_JOBID.log
