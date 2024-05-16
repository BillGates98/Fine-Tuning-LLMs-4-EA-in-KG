# Fine-Tuning LLMs for EA in KG :dizzy:

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

## 5. Final results

### 5.1. Evaluation on individual dataset :

![screenshot](result_on_individual.png)

### 5.2. Evaluation on merged dataset :

![screenshot](result_on_merged.png)
