#!/bin/bash

#PBS -P y89
#PBS -q gpuvolta
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=380GB
#PBS -l jobfs=400GB
#PBS -l walltime=48:00:00
#PBS -l wd

module load use.own
module load cuda/10.1
module load intel-mkl/2019.3.199
module load python3/3.7.4
module load sklearn/0.24.1
module load pytorch/1.5.1

python3 Cond_APOGEE_nfl_fulldim_fulldata_12labels.py $PBS_NCPUS > $PBS_JOBID.log
