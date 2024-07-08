from datasets import load_dataset
import pandas as pd
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import argparse
import time
import os


class GPTFineTuner:

    def __init__(self, suffix='', data_dir='', enable_trainer=True):
        self.data_dir = data_dir
        self.enable_trainer = enable_trainer
        self.suffix = suffix

    def load_data(self):
        dataset = load_dataset(
            "csv", data_dir=self.data_dir)  # '/content/person'
        return dataset

    def tokenizer_dataset(self):
        dataset = self.load_data()
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

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
        if 'merged' in self.suffix and os.path.exists(self.data_dir + '/../' + self.suffix):
            model = GPT2ForSequenceClassification.from_pretrained(
                self.data_dir + '/../' + self.suffix)
            from_saved = True
        else:
            model = GPT2ForSequenceClassification.from_pretrained(
                "gpt2", num_labels=2)
        training_args = TrainingArguments(
            output_dir="gpt2_trainer",
            # evaluation_strategy="epoch",
            per_device_train_batch_size=1,  # Reduce batch size here
            per_device_eval_batch_size=1,    # Optionally, reduce for evaluation as well
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
            f' {self.suffix} : Evaluation Time : {(time.time() - start_eval)} seconds ')
        if not from_saved:
            trainer.save_model(self.data_dir + '/../' + self.suffix)


if __name__ == "__main__":

    def arg_manager():
        parser = argparse.ArgumentParser()
        parser.add_argument("--base_dir", type=str, default="./outputs/")
        parser.add_argument("--suffix", type=str, default="model_gpt2")
        parser.add_argument("--enable_trainer", type=bool, default=True)
        return parser.parse_args()

    start = time.time()
    args = arg_manager()
    base_dir = args.base_dir
    suffix = args.suffix
    # "KnowledgeGraphsDataSet",
    benchmark_datasets = ["KnowledgeGraphsDataSet", "Yago-Wiki", "person", "restaurant", "anatomy",
                          "doremus", "SPIMBENCH_small-2019", "SPIMBENCH_large-2016"]

    for dir in benchmark_datasets[0:2]:
        print('Datasets on GPT-2 : ', dir)
        _model_name = dir if not "merged" in base_dir else args.suffix
        GPTFineTuner(suffix=_model_name, data_dir=base_dir+dir,
                     enable_trainer=args.enable_trainer).run()
        print('\n \n')

    print('Running Time : ', (time.time() - start), ' seconds ')
