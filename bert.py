from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
import evaluate
import numpy as np
import argparse
import time
import os


class BERTFineTuner:

    def __init__(self, suffix=None, data_dir='', enable_trainer=True):
        self.data_dir = data_dir
        self.enable_trainer = enable_trainer
        self.suffix = suffix

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
        from_saved = False
        model = None
        if 'merged' in self.suffix and os.path.exists(self.data_dir + '/../' + self.suffix):
            model = AutoModelForSequenceClassification.from_pretrained(
                self.data_dir + '/../' + self.suffix)
            from_saved = True
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                "google-bert/bert-base-cased", num_labels=2)
        training_args = TrainingArguments(
            output_dir="bert_trainer",
            # evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            num_train_epochs=2,
            weight_decay=0.01,
            gradient_accumulation_steps=4
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=eval,
            compute_metrics=self.compute_metrics,

        )
        if not from_saved:
            trainer.train()

        start_eval = time.time()
        trainer.evaluate()
        print(
            f' {self.suffix} : Evaluation Time : {time.time() - start_eval} seconds ')
        if not from_saved:
            trainer.save_model(self.data_dir + '/../' + self.suffix)


if __name__ == "__main__":
    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--base_dir", type=str, default="./outputs/")
        parser.add_argument("--suffix", type=str, default="model_bert")
        parser.add_argument("--enable_trainer", type=bool, default=True)
        return parser.parse_args()

    args = arg_manager()

    base_dir = args.base_dir  # "./outputs/merged/"
    suffix = args.suffix
    # ,
    benchmark_datasets = ["KnowledgeGraphsDataSet", "Yago-Wiki", "person", "restaurant",
                          "anatomy", "doremus", "SPIMBENCH_small-2019", "SPIMBENCH_large-2016"]

    for dir in benchmark_datasets[0:2]:
        print('Datasets on BERT : ', dir)
        _model_name = dir if not "merged" in base_dir else suffix
        BERTFineTuner(suffix=_model_name, data_dir=base_dir+dir,
                      enable_trainer=args.enable_trainer).run()
        print('\n \n')
