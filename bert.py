from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
import evaluate
import numpy as np
import argparse


class BERTFineTuner:

    def __init__(self, data_dir='', enable_trainer=True):
        self.data_dir = data_dir
        self.enable_trainer = enable_trainer

    def load_data(self):
        dataset = load_dataset("csv", data_dir=self.data_dir)
        return dataset

    def tokenizer_dataset(self):
        dataset = self.load_data()
        tokenizer = AutoTokenizer.from_pretrained(
            "google-bert/bert-base-cased")
        # tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            return tokenizer(examples["input"], padding="max_length", truncation=True)
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        return tokenized_datasets

    def splitting(self):
        tokenized_datasets = self.tokenizer_dataset()
        # .select(range(int(df1.shape[0]*0.7)))
        small_train_dataset = tokenized_datasets["train"]
        # .select(range(int(df1.shape[0]*0.3)))
        small_eval_dataset = tokenized_datasets["test"]
        return small_train_dataset, small_eval_dataset

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        metric1 = evaluate.load("accuracy")
        metric2 = evaluate.load("precision")
        metric3 = evaluate.load("recall")
        metric4 = evaluate.load("f1")
        return {"precision": metric2.compute(predictions=predictions, references=labels),
                "recall": metric3.compute(predictions=predictions, references=labels),
                "f1": metric4.compute(predictions=predictions, references=labels),
                "accuracy": metric1.compute(predictions=predictions, references=labels)}

    def run(self):
        train, eval = self.splitting()
        model = AutoModelForSequenceClassification.from_pretrained(
            "google-bert/bert-base-cased", num_labels=2)
        training_args = TrainingArguments(
            output_dir="bert_test_trainer",
            # evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=2,
            weight_decay=0.01,
            gradient_accumulation_steps=4
        )

        if self.enable_trainer:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train,
                eval_dataset=eval,
                compute_metrics=self.compute_metrics,

            )
            trainer.train()
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=None,
                eval_dataset=eval,
                compute_metrics=self.compute_metrics,

            )
        trainer.evaluate()


if __name__ == "__main__":
    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--base_dir", type=str, default="./outputs/")
        parser.add_argument("--enable_trainer", type=bool, default=True)
        return parser.parse_args()

    args = arg_manager()

    base_dir = args.base_dir  # "./outputs/merged/"
    benchmark_datasets = ["person", "restaurant", "anatomy",
                          "doremus", "SPIMBENCH_small-2019", "SPIMBENCH_large-2016"]

    for dir in benchmark_datasets:
        print('Datasets  : ', dir)
        BERTFineTuner(data_dir=base_dir+dir,
                      enable_trainer=args.enable_trainer).run()
        print('\n \n')
