#!/bin/bash
#SBATCH --job-name=b_amr
#SBATCH --output=b_ablation.txt
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --mem 16000
#SBATCH --nodelist=gpu08
#SBATCH --qos=batch

# Add ICL-Slurm binaries to path
GPU08SCRATCHPATH=/remote/gpu08/scratch/meier/
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
source ~/my_gpu09_venv/bin/activate
srun python3 -u -m joeynmt train 1c_ablation.yaml
#srun python3 -u -m joeynmt train ~/MeaningRep/gru_shortened_800.yaml  # BEst: tf_shortened_800.yaml #tf_shortened_even_smaller.yaml  # example job step
deactivate