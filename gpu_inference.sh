#!/bin/bash
#SBATCH --job-name=1b_pred
#SBATCH --output=1b_preds.txt #preds_1smatch.txt
#SBATCH --mail-user=meier@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL
#SBATCH --partition=students
#SBATCH --gres=gpu:1
#SBATCH --mem 16000
#SBATCH --nodelist=gpu09
#SBATCH --qos=batch

# Add ICL-Slurm binaries to path
#GPU08SCRATCHPATH=/remote/gpu08/scratch/meier/
#PATH=/opt/slurm/bin:$PATH

# JOB STEPS
source venv/bin/activate
#srun python3 -u -m joeynmt train ~/MeaningRep/tf.yaml  # example job step

#srun python3 -u -m joeynmt test ~/MeaningRep/models/tf/config.yaml --output_path ~/MeaningRep/models/tf/predictions
#python3 -m joeynmt translate ~/MeaningRep/models/tf/config.yaml < ~/MeaningRep/test/test.src > predictions.txt
#python3 -m joeynmt translate ~/MeaningRep/models/tf_no_shuffle/config.yaml < ~/MeaningRep/test_no_shuffle/test.src > predictions_no_shuffle.txt

#python3 -m joeynmt translate ~/MeaningRep/models/tf_short_small/config.yaml < ~/MeaningRep/test_short/test.src > prediction_short_small.txt

#python3 -m joeynmt translate ~/MeaningRep/models/tf_short_800/config.yaml < ~/MeaningRep/test_short/test.src > prediction_short_800.txt

# 1 Smatch
#python3 -m joeynmt translate ~/MeaningRep/models/tf_short_800/config.yaml < ~/MeaningRep/algorithm-synthesis-for-smatch/test.src > predictions_1_smatch.txt


#python3 -m joeynmt translate ~/MeaningRep/models/tf_ablation_800/config.yaml < ~/MeaningRep/test_ablation/test.src > ablation_predictions.txt

#1c
#python3 -m joeynmt translate ~/MeaningRep/models/new_ablation_800/config.yaml < ~/MeaningRep/test_ablation/test.src > 1c_ablation_predictions.txt

python3 -m joeynmt translate ~/MeaningRep/models/1b_ablation/config.yaml < ~/MeaningRep/1b_test/test.src > 1b_predictions.txt

deactivate