import os, sys
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import random
import pandas as pd
import argparse
import json
import ast
from tqdm import tqdm

import torch
import transformers
from datasets import load_dataset, load_metric, ClassLabel, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import pipeline, set_seed
from torchsummary import summary


class NMT(object):
    def __init__(self, opt):
        super(NMT, self).__init__()

        self.prefix = opt.prefix
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
        self.train = opt.train
        self.push_to_hub = opt.push_to_hub
        self.test_save_path = opt.test_save_path
        self.hub_model_id = opt.hub_model_id
        self.train_path = opt.train_path
        self.val_path = opt.val_path
        self.test_path = opt.test_path

        self.model_checkpoint = opt.model_checkpoint
        self.tokenizer_checkpoint = opt.tokenizer_checkpoint
        self.model_name = self.model_checkpoint.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_checkpoint, model_max_length=opt.max_input_length)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.dataset = None
        self.metric = None

    def load_data(self):
        self.metric = load_metric(self.metric_name)

        if not self.train:
            self.dataset = load_dataset(self.dataset_name, "{}-{}".format(self.source_lang, self.target_lang))
            self.dataset["train"] = self.dataset["train"].remove_columns(["date", "docIdx", "uid", "utcTimestamp", "rating", "status", "uid1LogInTime", "uid1LogOutTime", "uid1response", "uid2response", "user2_id", "whoSawDoc", "wikiDocumentIdx"])
            self.dataset["validation"] = self.dataset["validation"].remove_columns(["date", "docIdx", "uid", "utcTimestamp", "rating", "status", "uid1LogInTime", "uid1LogOutTime", "uid1response", "uid2response", "user2_id", "whoSawDoc", "wikiDocumentIdx"])
            self.dataset["test"] = self.dataset["test"].remove_columns(["date", "docIdx", "uid", "utcTimestamp", "rating", "status", "uid1LogInTime", "uid1LogOutTime", "uid1response", "uid2response", "user2_id", "whoSawDoc", "wikiDocumentIdx"])
        else:
            tdf = pd.read_csv(self.train_path)
            tdf.drop('Unnamed: 0', axis=1, inplace=True)
            tds = Dataset.from_pandas(tdf)
            vdf = pd.read_csv(self.val_path)    # val without BT ofcourse :)
            vdf.drop('Unnamed: 0', axis=1, inplace=True)
            ttdf = pd.read_csv(self.test_path) # test without BT ofcourse :)
            ttdf.drop('Unnamed: 0', axis=1, inplace=True)
            tds = Dataset.from_pandas(tdf)
            vds = Dataset.from_pandas(vdf)
            ttds = Dataset.from_pandas(ttdf)
            self.dataset = DatasetDict()
            self.dataset['train'] = tds
            self.dataset['validation'] = vds
            self.dataset['test'] = ttds

        print('1. data: ', self.dataset)
        print('2. a sample: ', self.dataset["train"][0])
        # print('3. description: ', self.dataset['train'].description)
        # print('4. Citation: ', self.dataset['train'].citation)      # copy-paste to report :P
        # print('5. metric: ', self.metric)
        # print('6. More samples:\n', self.show_random_samples(self.dataset["train"]))

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
        if self.train:
            inputs = [ast.literal_eval(ex)['prefix'] + ast.literal_eval(ex)['input'] for ex in examples["translation"]]
            targets = [ast.literal_eval(ex)['target'] for ex in examples["translation"]]
        else:
            inputs = [self.prefix + ex[self.source_lang] for ex in examples["translation"]]
            targets = [ex[self.target_lang] for ex in examples["translation"]]

        model_inputs = self.tokenizer(inputs, padding="longest", max_length=self.max_input_length, truncation=True)
        # print('model_inputs: ', model_inputs)
        # Setup the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def finetune(self):

        tokenized_datasets = self.dataset.map(self.preprocess_function, batched=True)
        print(tokenized_datasets)

        print('Model parameters: ', self.count_parameters())
        
        train_args = Seq2SeqTrainingArguments(
            f"{self.model_name}-finetuned-{self.source_lang}-to-{self.target_lang}", # set save directory name
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=64//self.batch_size,
            weight_decay=self.weight_decay,
            save_total_limit=self.save_total_limit,
            num_train_epochs=self.num_train_epochs,
            predict_with_generate=self.predict_with_generate,
            fp16=self.fp16,
            load_best_model_at_end=True, # Whether to load the best model found at each evaluation.
            metric_for_best_model="loss", # Use loss to evaluate best model.
            hub_model_id=self.hub_model_id,
            push_to_hub=self.push_to_hub    
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
        
        if self.train:
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

            if self.push_to_hub:
                trainer.push_to_hub(tags= "translation" , commit_message= "Training complete" )
        else:
            self.model.eval()
            result = trainer.evaluate(max_length=self.max_target_length)
            print("Evaluation: cmu val: ", result)
            torch.cuda.empty_cache()
            trainer.eval_dataset = tokenized_datasets["test"]
            result = trainer.evaluate(max_length=self.max_target_length)
            print("Evaluation: cmu test: ", result)
            torch.cuda.empty_cache()

        return None
    
    def generate_batch(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]
    
    def lince_inference(self):
        # for LinCE submission
        task_prefix = "translate English to Hinglish: "

        tdf = pd.read_csv('./data/hi_en/CALCS_hinglish_test_mt_en-hi_en.csv')
        tdf.drop('Unnamed: 0', axis=1, inplace=True)

        inputs = [ast.literal_eval(ex)['prefix'] + ast.literal_eval(ex)['input'] for ex in tdf["translation"]]
        gen = self.generate_batch(inputs, self.batch_size*2)
        output = []
        for input_batch in tqdm(gen, total=len(inputs)//(self.batch_size*2)+1):
            input_batch = self.tokenizer(input_batch, padding="longest", max_length=self.max_input_length, truncation=True, return_tensors="pt").to(self.device)
            self.model.eval()
            output_sequences = self.model.generate(input_ids=input_batch["input_ids"],
                                                attention_mask=input_batch["attention_mask"],
                                                do_sample=False, num_beams=20, num_beam_groups=5)
            
            output_batch = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            # print(output_batch[:1])
            output.extend(output_batch)

        fp = open(self.test_save_path, 'w+')
        for sentence in output:
            fp.write(sentence+"\n")
        fp.close()


def main(opt):
    pipeline = NMT(opt)
    pipeline.load_data()
    pipeline.finetune()

    # for LinCE submission
    if opt.prefix == "translate English to Hinglish: ":
        pipeline.lince_inference()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=bool, default=False, help='To train or do inference')
    parser.add_argument('--push_to_hub', type=bool, default=False, help='To train or do inference')

    parser.add_argument('--hub_model_id', type=str, default="sayanmandal/t5-small_6_3-test", help='set HuggingFace model ID')
    
    parser.add_argument('--prefix', type=str, default="translate Hinglish to English: ", help='set prefix')
    parser.add_argument('--source_lang', type=str, default="hi_en", help='set source language for eg. en')
    parser.add_argument('--target_lang', type=str, default="en", help='set target language for eg. hi_en')

    parser.add_argument('--train_path', type=str, default="./data/en/train_mt_hi_en-en.csv", help='train split path')
    parser.add_argument('--val_path', type=str, default="./data/en/val_mt_hi_en-en.csv", help='val split path')
    parser.add_argument('--test_path', type=str, default="./data/en/cmu_hinglish_dog_test_mt_hi_en-en.csv", help='val split path')
    parser.add_argument('--test_save_path', type=str, default="./mt_eng_hinglish.txt", help='lince test save path')

    parser.add_argument('--dataset_name', type=str, default="cmu_hinglish_dog", help='set dataset name for eg. cmu_hinglish_dog')
    parser.add_argument('--metric_name', type=str, default="sacrebleu", help='set metric')
    
    parser.add_argument('--model_checkpoint', type=str, default="./t5_small_6_3", help='set model checkpoint, eg. google/mt5-small')
    parser.add_argument('--tokenizer_checkpoint', type=str, default="./t5_small_6_3", help='set tokenizer checkpoint, eg. google/mt5-small')
    
    parser.add_argument('--num_train_epochs', type=int, default=50, help="set num epochs")
    parser.add_argument('--save_total_limit', type=int, default=3, help="set save total limit")
    parser.add_argument('--lr', type=float, default=1e-4, help="set learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="set weight decay")
    parser.add_argument('--batch_size', type=int, default=8, help="set batch size")
    parser.add_argument('--num_samples', type=int, default=5, help="set number of samples of dataset to display")
    parser.add_argument('--max_input_length', type=int, default=512, help="set max input length")
    parser.add_argument('--max_target_length', type=int, default=512, help="set max target length")
    opt = parser.parse_args()
    main(opt)