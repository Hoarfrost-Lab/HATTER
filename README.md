
![hatter-logo](https://github.com/Hoarfrost-Lab/HATTER/blob/main/assets/HATTER-logo.png)
# HATTER: Human-in-the-loop Adaptive Toolkit for Transferable Enzyme Representations

This repository serves as the official release 1.0 of the multi-agent system proposed by Babjac and Hoarfrost et. al. as described in "Active Learning in Wonderland — A Toolkit for Curious Machines and Curious Biologists". 

![hatter-pipeline](https://github.com/Hoarfrost-Lab/HATTER/blob/main/assets/HATTER-pipeline.png)

**Additional Features**

- Integrated option of correlating active learning scores with scores from exprimental design pre-analysis (or any values of interest to the researcher)
- Select query/active learning score by ID not EC
- Integrated active learning pre-training
- Simulation mode (for researchers to be able to decide best params for experiment without trial/error)
- Integrated standard two layer classifier (can be easily modified to adapt to a wide range of tasks)

**Bug Fixes and Improvements (since DARPA completion)**

- Bayesian / BALD  
  - Fixed MC dropout integration in update  
  - Fixed multi-learner shape issue  

- QBC (Query by Committee)  
  - Fixed parsing to work column-wise instead of row-wise  

- Bio-inspired, BADGE, Typiclust  
  - Integrated memory optimization to reduce CPU usage  
  - Still very slow: cannot handle more than ~1000 proteins in pool file  

- CLEAN Integration  
  - Fixed validation bug

- Model Update
  - Explicitly recomputes distmaps if not specified a recomputed one  
  - Automatically augments incorrect proteins by adding duplicates for update  
  - Copies precomputed embeddings from pool directory to train directory during update  
  - Fixed logic in `train/pool ec_id/id_ec` maps  
  - Samples positive/negative points from `train_ids` (if possible), otherwise defaults to `pool_ids` during update  

**Inputs and Outputs**

The required inputs are the same as the CLEAN architecture. It requires a train_csv_file and a pool_csv_file. Validation and testing files are optional. They should follow the following format (in .tsv format):

| Entry  | EC number | Sequence |
|--------|-----------|----------|
| P0AB74 |           | MSIISTKY... |
| Q8WZJ7 |           | MALLLEGTSL... |


Notes on inputs: 
1. The pool_csv_file should have blank EC number column as shown above (but column must still exist and will be annotated by the software), whereas train, validation and test datasets should contain EC labels separated by “;” (refer to CLEAN)
2. The pool_csv_file can already contain labels (if pre-annotated or experimental) and the "--labeled" flag should be specified so they do not get overwritten.


To run the first step of the software, you must call init mode (once). Here is an example command:

```bash
python3 src/driver.py \
  --mode init \
  --active_type uncertainty_sampling \
  --train_csv_path ./example_data/split100_tiny.csv \
  --batch_size 128 \
  --perform_pretraining \
  --num_train_epochs 1 \
  --save_path ./example_init_mode/ \
  --generate_plots \
  --checkpoint_and_eval \
  --pool_csv_path ./example_data/split100_tiny_no_labels.csv \
  --num_instances 8 \
  --round_number 0 \
  --valid_csv_path ./example_data/split100_tiny.csv \
  --test_csv_paths ./example_data/split100_tiny.csv
 ```
The above command will perform pre-training using "--train_csv_path" but a pre-trained model can also be passed instead using "--model_load_path".

The expected outputs in init mode are in the same format as the pool_csv_file but contains an extra column “Result” which is blank and should be filled out by the researcher during the experimental process. This file will be called “infer_ids.tsv” and is written to the "--save_path" specified by the user. We include an example below:

| Entry  | EC number | Sequence | Result |
|--------|-----------|----------|--------|
| P0AB74 | 4.1.2.40  | MSIISTKY... |        |
| Q8WZJ7 | 6.3.4.3   | MALLLEGTSL... |        |


Additionally, the software can produce validation/text metrics in json format (if specified) and plots of active learning (if specified) – written to "--save_path".

Once the “Result” column is filled in by the experiment, the update mode can be run. This mode takes the same parameters as init mode as well as an optional flag “--update_and_requery” which specifies if you would like additional id’s to be selected from the pool_csv_file. An example command is:

```bash
python3 src/driver.py \
  --mode update \
  --active_type uncertainty_sampling \
  --train_csv_path ./example_data/split100_tiny.csv \
  --save_path ./example_update_mode/ \
  --generate_plots \
  --checkpoint_and_eval \
  --pool_csv_path ./example_data/split100_tiny_labeled_experimental_result.csv \
  --num_instances 8 \
  --round_number 1 \
  --valid_csv_path ./example_data/split100_tiny.csv \
  --test_csv_paths ./example_data/split100_tiny.csv \
  --precomputed
  --model_load_path ./models/modelweights.pt
```

Note: the pool_csv_path should now have True/False in the “Result” column, where True represents the EC number being successful in the experiment and False represents unsuccessful. For example:

| Entry  | EC number | Sequence | Result |
|--------|-----------|----------|--------|
| P0AB74 | 4.1.2.40  | MSIISTKY...| True  |
| Q8WZJ7 | 6.3.4.3   | MALLLEGTSL... | False |


The update mode will write several files to “--save_path”, including: 
- CLEAN_final.pth – the updated CLEAN model weights
- training_metrics.json – train (and optional: validation) loss and additional metrics
- distance_map_recomputed (directory) – updated distmaps the researcher can choose to use in future rounds
- infer_ids.tsv (optional) – new id’s for the next round (if “update_and_requery” flag is specified)
additional plots and metrics (optional):
- pca*.png – pca plots of uncertainty scores (if “generate_plots” flag is specified)
- *_metrics.json – metrics for test datasets (if “test_csv_path” is specified)
- lossplot.png – lossplot (if num_epochs > 1 and “checkpoint_and_eval” or “generate_plots” is specified)

The above steps can be repeated for any number of rounds assuming appropriate pool data. The software does not handle the following edge cases:
1. Less ids in pool_csv_file than “num_instances” (the requested number of ids for that round)
2. Less ecs in pool_csv_file than the “num_instances” (the requested number of ids for that round) – this version of software queries by EC not ID but is fixed in later versions
3. Errors in sequence or invalid amino acids
4. Errors in file format including missing EC numbers or True/False values


**Additional Modes**

- train: pre-training of a CLEAN model.
- inference: inference on test_data_paths given a specified CLEAN model path.
- simulation: can simulate rounds of experiment (same as AL_simulation results in Q4-experimental-design-agent). Please refer to Q4-experimental-design-agent for examples of running simulations.


**Notes on Environment/Install**

- Our model training was completed on CUDA=12.1 and Python 3.11. We include a environment.yml with the exact package installations used here.

- This package is dependent on both dal-toolbox (https://github.com/dhuseljic/dal-toolbox) and CLEAN (https://github.com/tttianhao/CLEAN/tree/main) which CANNOT be installed with pip/conda. We have cloned these repo's under "dal_toolbox_app" and "clean_app" in this repo to maintain version. You can follow the install instructions using the respective repo pages to install them directly.

- CLEAN is dependent on ESM. The package "fair-esm" is installed as part of the conda environment, but the ESM repo must also be cloned into the main folder of this repo in order to function properly (run "git clone https://github.com/facebookresearch/esm.git" outside the "scripts" folder). 

- We strongly recommend using conda for install and package dependency management. If using different version of CUDA/Python we recommend updating the environment.yml file or starting from pytorch locally (https://pytorch.org/get-started/locally/) and installing all other requiements with conda/pip afterwards. 

- We note all models were able to be fit onto a single Nvidia A100-80GB GPU and train in ~24 hours or less using an appropriate batch size. If running into memory issues we recommend lowering the batch size or shortening the sequence length of a given model.

- If additional problems with environment occur please reach out to the authors and we will be happy to assist you!


