# Smaragd - Synthesized sMatch for accurate and rapid AMR graph distance
Offical repository of [Smaragd](https://arxiv.org/abs/2203.13226).

Directories A, B and C contain the evaluation scripts for the specific model type. In general, the scripts are similar 
to each other, containing only minor changes. Best performing model is model A, configuration and checkpoint
are provided.

## Training and Inference

For Training and Inference a virtual environment with Python 3.8.10 was used. JoeyNMT 1.4 with Torch 1.9 and 
Torchtext 0.10.0 was used. A requirements file (inference_req.txt) for recreating this virtual environment is included.
 
For further usage about joey-nmt, please consider the offical [repos](https://github.com/joeynmt/joeynmt).

JoeyNMT works with yaml files, defining the models parameters. We provide the configuration of the A-model, as
well the checkpoint used for inference. In order to perform inference on your machine, the parameter model_dir
in the yaml file must be adapted to the location on your machine.

Example for training
```
python3 -u -m joeynmt train ~/AMR_ablation/config.yaml
```

Example for inference of joeynmt:
```
python3 -m joeynmt translate ~/MeaningRep/models/tf/config.yaml < ~/MeaningRep/test/test.src > predictions.txt
```
We call the joeynmt module via python3 -m. Joeynmt is advised to translate using the config of the model 
defined in the yaml file. The input is defined after the < character. The outputfile is declared in the last 
part as predictions.txt.

## Evaluation

Evaluation requires at least Python 3.7. An requirements file is again included. 
To recreate the results from the A-Model in row 7, run the command below:
```
python3 eval_ablation.py --test_source a_test.src --gold_align a_test.tgt --pred_align 00305000.hyps.test --first_amr 
gpla_first_test_set_reduced.txt --second_amr gpla_second_test_set_reduced.txt
```
To recreate the normal baseline, add `--baseline` to the command. For the smarter baseline add `--baseline`
 and `--smart_baseline`.  To perform multisentence evaluation add the parameter `--multisentence`
 
In order to perform a constrained study through limiting the number of variables, three parameters are necessary:

- constrained_study: Boolean value if a constrained study should be done or not
- lower_bound: Lower bound of variables, eg. 35
- upper_bound: Upper bound of variables, eg. 130

# Results

Baseline results are averaged over 10 runs.

| Model  | F1 Gold | F1 Pred | Prec-Gold | Prec-Pred | Rec-Gold | Rec-Pred | Alignment Accuracy
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| A-Baseline | 77.46 | 13.60  | 79.69 | 14.17 | 76.66 | 13.43 | 0.3676 |
| B-Baseline | 77.46| 13.51 | 79.69 | 14.09| 76.66 | 13.32 | 0.3665 |
| C-Baseline | 77.46| 13.41 | 79.69 | 13.97| 76.66 | 13.25 | 0.4490 |
| A-Baseline (smarter) | 77.46 | 68.86  | 79.69  | 70.83  | 76.66  | 68.16  | 0.6500 |
| B-Baseline (smarter) | 77.46 | 68.86   | 79.69  | 70.83  | 76.66  | 68.16  | 0.6500 |
| C-Baseline (smarter) | 77.46 | 68.86   | 79.69  | 70.83  | 76.66  | 68.16 | 0.5594 |
| A-Model | 77.46| 76.37 | 79.69 | 78.51 | 76.66 | 75.59 | 0.8848 |
| B-Model | 77.46| 64.48 | 79.69 | 66.16 | 76.66 | 63.98 | 0.5677 |
| C-Model | 77.65| 39.00 | 79.84 | 39.64 | 76.93 | 38.93 | 0.3958 |

| Model  | STD Gold | STD Pred | Pearson F1 Scores | 
| ------------- | ------------- | ------------- | ------------- |
| A-Baseline | 0.2478  | 0.2107  | (0.22920455999157135, 1.3714024019852316e-16) |
| B-Baseline | 0.2478 | 0.2076| (0.22346729673814497, 9.517628413067954e-16) |
| C-Baseline | 0.2497  | 0.2079 | (0.22220299314949524, 1.5428378321805351e-16) | 
| A-Baseline (smarter) | 0.2478  | 0.2768 | (0.8814932094234014, 0.0)  |
| B-Baseline (smarter) | 0.2478  | 0.2768 | (0.8814932094234014, 0.0)  | 
| C-Baseline (smarter) | 0.2497 | 0.2768 | (0.8814932094234014, 0.0)   |
| A-Model | 0.2478 | 0.2574 | (0.9836660698339711, 0.0) | 
| B-Model | 0.2478 | 0.3258| (0.7496214337800232, 8.983217705385425e-271) | 
| C-Model | 0.2498 | 0.3618 | (0.5281452393797889, 1.630686585891851e-108) | 





