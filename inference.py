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
import time

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
        self.batch_size = opt.batch_size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model_checkpoint = opt.model_checkpoint
        self.tokenizer_checkpoint = opt.tokenizer_checkpoint
        self.model_name = self.model_checkpoint.split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_checkpoint, model_max_length=opt.max_input_length)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
    
    def generate_batch(self, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]

    def predict_sentence(self, input_sequence_list):
        start = time.time()

        inputs = [self.prefix + ex for ex in input_sequence_list]
        
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
        end = time.time()
        print("\n\nInput:\n{}".format(inputs))
        print("Translated:\n{}".format(output))
        print("Time taken: {:.02f} seconds".format(end-start))
        return output
    
    def predict_csv(self, input_path):
        start = time.time()

        tdf = pd.read_csv(input_path)
        tdf.drop('Unnamed: 0', axis=1, inplace=True)

        inputs = [ast.literal_eval(ex)['prefix'] + ast.literal_eval(ex)['input'] for ex in tdf["translation"]]
        x = [ast.literal_eval(ex)['input'] for ex in tdf["translation"]]
        y = [ast.literal_eval(ex)['target'] for ex in tdf["translation"]]
        
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
        end = time.time()
        print("Time taken: {:.02f} seconds".format(end-start))

        assert len(x) == len(y) == len(output)

        data = []
        for i in range(len(x)):
            data.append([x[i], y[i], output[i]])
        
        df = pd.DataFrame(data, columns=["source", "target", "predicted"])
        df.to_csv('prediction_hi_en-en.csv')

        return output


def main(opt):
    pipeline = NMT(opt)
    input = [opt.input]
    output = pipeline.predict_sentence(input)
    
    output = pipeline.predict_csv('./data/en/CALCS_hinglish_dev_mt_hi_en-en.csv')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default="translate Hinglish to English: ", help='set prefix')
    parser.add_argument('--batch_size', type=int, default=16, help="set batch size")
    parser.add_argument('--max_input_length', type=int, default=512, help="set max input length")
    parser.add_argument('--max_target_length', type=int, default=512, help="set max target length")

    parser.add_argument('--input', type=str, default="Ye movie kis baare main hai?", help='sentence to translate')

    parser.add_argument('--model_checkpoint', type=str, default="sayanmandal/t5-small_6_3-hi_en-en_mix", help='set model checkpoint, eg. google/mt5-small')
    parser.add_argument('--tokenizer_checkpoint', type=str, default="sayanmandal/t5-small_6_3-hi_en-en_mix", help='set tokenizer checkpoint, eg. google/mt5-small')

    opt = parser.parse_args()
    main(opt)