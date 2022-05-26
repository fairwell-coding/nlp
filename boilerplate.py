import os, sys
os.environ["WANDB_DISABLED"] = "true"
import numpy as np
import random
import pandas as pd
import argparse

import torch
import transformers
from datasets import load_dataset, load_metric, ClassLabel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import pipeline, set_seed
from torchsummary import summary

from  huggingface_hub  import  notebook_login
notebook_login()


class NMT(object):
    def __init__(self, opt):
        super(NMT, self).__init__()

        self.prefix=opt.prefix
        self.max_input_length = opt.max_input_length
        self.max_target_length = opt.max_target_length
        self.source_lang = opt.source_lang
        self.target_lang = opt.target_lang
        self.dataset_name = opt.dataset_name
        self.metric_name = opt.metric_name
        self.num_samples = opt.num_samples
        self.batch_size = opt.batch_size
        self.lr = opt.lr
        self.weight_decay = opt.weight_decay
        self.save_total_limit = opt.save_total_limit
        self.num_train_epochs = opt.num_train_epochs
        self.predict_with_generate = True
        self.fp16 = True
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"


        self.model_checkpoint = opt.model_checkpoint
        self.model_name = self.model_checkpoint.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint, model_max_length=42)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint).to(self.device)

        self.dataset = None
        self.metric = None

    def load_data(self):
        self.dataset = load_dataset(self.dataset_name, "{}-{}".format(self.source_lang, self.target_lang))
        self.metric = load_metric(self.metric_name)

        print('1. data: ', self.dataset)
        print('2. a sample: ', self.dataset["train"][0])
        print('3. description: ', self.dataset['train'].description)
        print('4. Citation: ', self.dataset['train'].citation)      # copy-paste to report :P
        print('5. metric: ', self.metric)
        print('6. More samples:\n', self.show_random_samples(self.dataset["train"]))
        print('7. Dataset: ', self.dataset)

    def show_random_samples(self, dataset):
        assert self.num_samples <= len(dataset), "Your request exceeds dataset! :("
        picks = []
        for _ in range(self.num_samples):
            pick = random.randint(0, len(dataset)-1)
            while pick in picks:
                pick = random.randint(0, len(dataset)-1)
            picks.append(pick)
        
        df = pd.DataFrame(dataset[picks])
        for column, typ in dataset.features.items():
            if isinstance(typ, ClassLabel):
                df[column] = df[column].transform(lambda i: typ.names[i])
        
        return df

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def preprocess_function(self, examples):
        inputs = [self.prefix + ex[self.source_lang] for ex in examples["translation"]]
        targets = [ex[self.target_lang] for ex in examples["translation"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def finetune(self):

        tokenized_datasets = self.dataset.map(self.preprocess_function, batched=True)

        print('Model parameters: ', self.count_parameters())
        
        train_args = Seq2SeqTrainingArguments(
            f"{self.model_name}-finetuned-{self.source_lang}-to-{self.target_lang}",
            # evaluation_strategy = "steps",
            # eval_steps=1000,
            evaluation_strategy = "epoch",
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=4,
            weight_decay=self.weight_decay,
            save_total_limit=self.save_total_limit,
            num_train_epochs=self.num_train_epochs,
            predict_with_generate=self.predict_with_generate,
            fp16=self.fp16,
            push_to_hub=True    
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)

        trainer = Seq2SeqTrainer(
                                    self.model,
                                    train_args,
                                    train_dataset=tokenized_datasets["train"],
                                    eval_dataset=tokenized_datasets["validation"],
                                    data_collator=data_collator,
                                    tokenizer=self.tokenizer,
                                    compute_metrics=self.compute_metrics
                                )
        
        self.model.eval()
        result = trainer.evaluate(max_length=self.max_target_length)
        print("Before training, val: ", result)
        torch.cuda.empty_cache()
        trainer.eval_dataset = tokenized_datasets["test"]
        result = trainer.evaluate(max_length=self.max_target_length)
        print("Before training, test: ", result)
        torch.cuda.empty_cache()

        self.model.train()
        trainer.eval_dataset = tokenized_datasets["validation"]
        trainer.train()
        torch.cuda.empty_cache()

        self.model.eval()
        result = trainer.evaluate(max_length=self.max_target_length)
        print("After training, val", result)
        torch.cuda.empty_cache()

        trainer.eval_dataset = tokenized_datasets["test"]
        result = trainer.evaluate(max_length=self.max_target_length)
        print("After training, test", result)
        torch.cuda.empty_cache()

        trainer.push_to_hub(tags= "translation" , commit_message= "Training complete" )

        return None


def main(opt):
    pipeline = NMT(opt)
    pipeline.load_data()
    pipeline.finetune()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default="translate Hinglish to English: ", help='set prefix')
    parser.add_argument('--source_lang', type=str, default="hi_en", help='set source language for eg. en')
    parser.add_argument('--target_lang', type=str, default="en", help='set target language for eg. hi_en')
    parser.add_argument('--dataset_name', type=str, default="cmu_hinglish_dog", help='set dataset name for eg. cmu_hinglish_dog')
    parser.add_argument('--metric_name', type=str, default="sacrebleu", help='set metric')
    parser.add_argument('--model_checkpoint', type=str, default="t5-small", help='set model checkpoint, eg. rossanez/t5-small-finetuned-de-en-wd-01 or google/mt5-small')
    parser.add_argument('--num_train_epochs', type=int, default=100, help="set num epochs")
    parser.add_argument('--save_total_limit', type=int, default=3, help="set save total limit")
    parser.add_argument('--lr', type=float, default=2e-5, help="set learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="set weight decay")
    parser.add_argument('--batch_size', type=int, default=16, help="set batch size")
    parser.add_argument('--num_samples', type=int, default=5, help="set number of samples of dataset to display")
    parser.add_argument('--max_input_length', type=int, default=128, help="set max input length")
    parser.add_argument('--max_target_length', type=int, default=128, help="set max target length")
    opt = parser.parse_args()
    main(opt)