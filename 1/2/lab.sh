#!/bin/bash
#PBS -l select=1:ncpus=1:mpiprocs=1:mem=10000m,place=free:exclhost
#PBS -l walltime=00:05:00
cd $PBS_O_WORKDIR
MPI_NP=$(wc -l $PBS_NODEFILE | awk &#39;{ print $1 }&#39;)
mpirun -hostfile $PBS_NODEFILE -np $MPI_NP ./lab.exe