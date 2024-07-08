# Fine-Tuning LLMs and from-scratch training of KAN for EA in KG :dizzy:

> Install Python :snake: (>=3.8) and dependencies.

    python -m pip install -r ./requirements.txt

> Important to do :

    - Unzip the data in the 'inputs' directory and move them into the root of 'inputs' directory

    - If you don't want to re-run evaluations, just open the 'outputs' directory and read the results.

## 1. Data sampling : train and test data :

    $ ./job_sampling.sh

## 2. Fine-tuning on individual dataset

    $  ./job_individual_fine_tuning.sh

## 3. Merge Train sets of individual datasets

    $ ./job_merge_train_data.sh

## 4. Fine-tuning on merged dataset

    $ ./job_merged_fine_tuning.sh

## 5. Generalization on unseen data

    $ ./job_merged4_fine_tuning.sh

## [Checkpoints of fine-tuned models](https://drive.google.com/drive/folders/1a_WUu006b6tDVsaC9by9S2Ip0bXaPxk0?usp=sharing)

