#!/bin/bash
#SBATCH -J 2dpbc
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6G
#SBATCH -t 8:00:00
#SBATCH -p inferno
#SBATCH -o 2dpbc-%A_%a.out
cd $SLURM_SUBMIT_DIR 
printf -v j "%06d" ${SLURM_ARRAY_TASK_ID}

module load intel
module load intel-mpi
module load abaqus

WORKDIR=./${j}

mkdir -p "$WORKDIR"
cp -r micros_hetero.npy ./${j}/micros_hetero.npy
cp -r abaqus_array.py ./${j}/abaqus_array.py
cd "$WORKDIR" || exit -1

srun -N 1 -n 1 -c $SLURM_CPUS_PER_TASK --mem-per-cpu=$SLURM_MEM_PER_CPU --exclusive abaqus cae noGUI=main.py -- ${j}

