#!/bin/bash
#SBATCH --job-name=AMR_prepro
#SBATCH --output=prepro_C.txt
#SBATCH --mail-user=meier@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem 30000
#SBATCH --qos=batch

# Add ICL-Slurm binaries to path
PATH=/opt/slurm/bin:$PATH

# JOB STEPS
cd algorithm-synthesis-for-smatch/

# 17.01 Office J, Data for Model A and B

#srun python3 preprocess.py /home/students/meier/MeaningRep/amr-align-data/py3-Smatch-and-S2match/smatch/data/processed/combined_amr3.gpla_combined_amr3.txt_first.amr /home/students/meier/MeaningRep/amr-align-data/py3-Smatch-and-S2match/smatch/data/processed/combined_amr3.gpla_combined_amr3.txt_second.amr /home/students/meier/MeaningRep/amr-align-data/py3-Smatch-and-S2match/smatch/data/processed/combined_amr3.gpla_combined_amr3.txt_align-restarts10

# 17.01 Office J, Data for Model C

#srun python3 preprocess.py /home/students/meier/MeaningRep/amr-align-data/py3-Smatch-and-S2match/smatch/data/processed_non_ano/combined_amr3.gpla_combined_amr3.txt_first.amr /home/students/meier/MeaningRep/amr-align-data/py3-Smatch-and-S2match/smatch/data/processed_non_ano/combined_amr3.gpla_combined_amr3.txt_second.amr /home/students/meier/MeaningRep/amr-align-data/py3-Smatch-and-S2match/smatch/data/processed_non_ano/combined_amr3.gpla_combined_amr3.txt_align-restarts10










#srun python3 randomize_and_split.py /home/students/meier/MeaningRep/amr-align-data/py3-Smatch-and-S2match/smatch/data/processed_edit/reduced/combined_amr3.jamr_combined_amr3.txt_first.amr_tokenized_reduced.txt /home/students/meier/MeaningRep/amr-align-data/py3-Smatch-and-S2match/smatch/data/processed_edit/reduced/combined_amr3.jamr_combined_amr3.txt_first.amr_tokenized_reduced.txt /home/students/meier/MeaningRep/amr-align-data/py3-Smatch-and-S2match/smatch/data/processed_edit/reduced/combined_amr3.jamr_combined_amr3.txt_align-reduced_restarts1
