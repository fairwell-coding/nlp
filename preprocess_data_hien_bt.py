
import numpy as np
import pandas as pd
import re
import sys
import string
import enchant
import json
from random import shuffle
from glob import glob
from tqdm import tqdm

# INDIC-TRANSLITERATION
from indic_transliteration import detect
from indic_transliteration import sanscript
from indic_transliteration.detect import Scheme
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

import torch
import transformers
from datasets import load_dataset, load_metric, ClassLabel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import pipeline, set_seed
from torchsummary import summary
from transformers import MarianMTModel, MarianTokenizer

d = enchant.Dict("en_US")

################## Pre-Process Functions #######################
'''
Borrowed from https://github.com/piyushmakhija5/hinglishNorm/blob/master/dataPreprocessing.py
Thanks @piyushmakhija5 🥳🥳
'''

def regex_or(*items):
    r = '|'.join(items)
    r = '(' + r + ')'
    return r

def pos_lookahead(r):
    return '(?=' + r + ')'

def neg_lookahead(r):
    return '(?!' + r + ')'

def optional(r):
    return '(%s)?' % r

PunctChars = r'''[`'“".?!,:;]'''
Punct = '%s+' % PunctChars
Entity = '&(amp|lt|gt|quot);'

def to_lowerCase(text):
    '''
    Convert text to lower case alphabets
    '''
    return str(text).lower()

def trim(text):
    '''
    Trim leading and trailing spaces in the text
    '''
    return text.strip().strip('.')

def strip_whiteSpaces(text):
    '''
    Strip all white spaces
    '''
    text = re.sub(r'[\s]+', ' ', text)
    return text

def process_URLs(text):
    '''
    Replace all URLs in the  text
    '''
    UrlStart1 = regex_or('https?://', r'www\.')
    CommonTLDs = regex_or('com','co\\.uk','org','net','info','ca','biz','info','edu','in','au')
    UrlStart2 = r'[a-z0-9\.-]+?' + r'\.' + CommonTLDs + pos_lookahead(r'[/ \W\b]')
    UrlBody = r'[^ \t\r\n<>]*?'  # * not + for case of:  "go to bla.com." -- don't want period
    UrlExtraCrapBeforeEnd = '%s+?' % regex_or(PunctChars, Entity)
    UrlEnd = regex_or( r'\.\.+', r'[<>]', r'\s', '$')
    Url =   (optional(r'\b') +
            regex_or(UrlStart1, UrlStart2) +
            UrlBody +
    pos_lookahead( optional(UrlExtraCrapBeforeEnd) + UrlEnd))

    Url_RE = re.compile("(%s)" % Url, re.U|re.I)
    text = re.sub(Url_RE, ' ', text)

    # fix to handle unicodes in URL
    URL_regex2 = r'\b(htt)[p\:\/]*([\\x\\u][a-z0-9]*)*'
    text = re.sub(URL_regex2, ' ', text)
    return text

def apply_transliteration(text):
    '''
    Detect Language and Transliterate non-Roman script text into Roman script
    '''
    lang = detect.detect(str(text))
    if lang not in [Scheme.ITRANS, Scheme.HK, Scheme.SLP1, Scheme.IAST, Scheme.Velthuis, Scheme.Kolkata]:
        text = transliterate(text, getattr(sanscript, lang.upper()), sanscript.HK).lower()
    return text

def filter_alpha_numeric(text):
    '''
    Filter out words which have non-Alpha-numeric characters
    '''
    if text!=None:
        tokens = text.split()
        clean_tokens = [t for t in tokens if re.sub(r'[^\w\s]',' ',t)]
        return ' '.join(clean_tokens)
    else:
        return text

def remove_punctuations(text):
    '''
    Filter out punctuations from the sentence
    '''
    return re.sub('[%s]' % re.escape(string.punctuation), '', text)

def remove_non_ascii(text):
    '''
    Filter out words which are outside Ascii-128 character range
    '''
    if text!=None:
        return ''.join(i for i in text if ord(i)<128)
    else:
        return text

def remove_empty(df, columnName):
    '''
    Remove sentences which have become empty after applying all preprocessing functions
    '''
    df[columnName].replace('', np.nan, inplace=True)
    df[columnName].replace(r'^\s*$', np.nan, inplace=True)
    df.dropna(subset=[columnName], inplace=True)
    return df

############################ END #################################

######################## Driver Function #########################

def preprocess(df, columnName):
    print("Processing ..")
    print("Total length : ",len(df))
    # df[columnName] = df[columnName].apply(apply_transliteration)
    df[columnName] = df[columnName].apply(to_lowerCase)
    df[columnName] = df[columnName].apply(process_URLs)
    df[columnName] = df[columnName].apply(filter_alpha_numeric)
    df[columnName] = df[columnName].apply(remove_punctuations)
    df[columnName] = df[columnName].apply(remove_non_ascii)
    df[columnName] = df[columnName].apply(trim)
    df[columnName] = df[columnName].apply(strip_whiteSpaces)
    df = remove_empty(df, columnName)
    df = df.reset_index(drop=True)
    print("Processing  Complete !!")
    return df

############################ END ###############################



def translate(texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
    src_texts = [template(text) for text in texts]

    # Tokenize the texts
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    encoded = tokenizer(src_texts, return_tensors="pt", max_length=512, padding=True, truncation=True).to(device)
    
    # Generate translation using model
    translated = model.generate(**encoded)

    # Convert the generated tokens indices back into text
    translated_texts = tokenizer.batch_decode(translated, skip_special_tokens=True)
    
    return translated_texts

target_model_name1 = 'Helsinki-NLP/opus-mt-en-{}'.format("de")
target_tokenizer1 = MarianTokenizer.from_pretrained(target_model_name1)
target_model1 = MarianMTModel.from_pretrained(target_model_name1).cuda()

en_model_name1 = 'Helsinki-NLP/opus-mt-{}-en'.format("de")
en_tokenizer1 = MarianTokenizer.from_pretrained(en_model_name1)
en_model1 = MarianMTModel.from_pretrained(en_model_name1).cuda()

target_model_name2 = 'Helsinki-NLP/opus-mt-en-{}'.format("zh")
target_tokenizer2 = MarianTokenizer.from_pretrained(target_model_name2)
target_model2 = MarianMTModel.from_pretrained(target_model_name2).cuda()

en_model_name2 = 'Helsinki-NLP/opus-mt-{}-en'.format("zh")
en_tokenizer2 = MarianTokenizer.from_pretrained(en_model_name2)
en_model2 = MarianMTModel.from_pretrained(en_model_name2).cuda()

target_model_name3 = 'Helsinki-NLP/opus-mt-en-{}'.format("es")
target_tokenizer3 = MarianTokenizer.from_pretrained(target_model_name3)
target_model3 = MarianMTModel.from_pretrained(target_model_name3).cuda()

en_model_name3 = 'Helsinki-NLP/opus-mt-{}-en'.format("es")
en_tokenizer3 = MarianTokenizer.from_pretrained(en_model_name3)
en_model3 = MarianMTModel.from_pretrained(en_model_name3).cuda()

target_model_name4 = 'Helsinki-NLP/opus-mt-en-{}'.format("fr")
target_tokenizer4 = MarianTokenizer.from_pretrained(target_model_name4)
target_model4 = MarianMTModel.from_pretrained(target_model_name4).cuda()

en_model_name4 = 'Helsinki-NLP/opus-mt-{}-en'.format("fr")
en_tokenizer4 = MarianTokenizer.from_pretrained(en_model_name4)
en_model4 = MarianMTModel.from_pretrained(en_model_name4).cuda()

model_dict = {"en_de": [target_tokenizer1, target_model1, en_tokenizer1, en_model1],
            "en_zh": [target_tokenizer2, target_model2, en_tokenizer2, en_model2],
            "en_es": [target_tokenizer3, target_model3, en_tokenizer3, en_model3],
            "en_fr": [target_tokenizer4, target_model4, en_tokenizer4, en_model4]
            }

def back_translate(texts, source_lang="en", target_lang="fr"):

    target_tokenizer, target_model, en_tokenizer, en_model = model_dict["{}_{}".format(source_lang, target_lang)]
    
    # Translate from source to target language
    fr_texts = translate(texts, target_model, target_tokenizer, 
                         language=target_lang)

    # Translate from target language back to source language
    back_translated_texts = translate(fr_texts, en_model, en_tokenizer, 
                                      language=source_lang)
    
    return back_translated_texts


def generate_batch(lst1, lst2, batch_size):
    for i in range(0, len(lst1), batch_size):
        yield [lst1[i : i + batch_size], lst2[i : i + batch_size]]

def cmu_hinglish_dog():
    dataset = load_dataset('cmu_hinglish_dog', "hi_en-en")
    splits = ['train', 'validation']
    for sp in splits:
        df = pd.DataFrame(dataset[sp])
        df.drop(df.columns.difference(['translation']), 1, inplace=True)
        print(df.head())
        df.to_csv('./data/cmu_hinglish_dog_{}.csv'.format(sp))
        eng = [ex['en'] for ex in df['translation']]
        hi_en = [ex['hi_en'] for ex in df['translation']]
        
        data = []
        gen = generate_batch(hi_en, eng, 4)
        for hinglish, english in tqdm(gen, total=len(eng)//4+1):
            aug1_texts = back_translate(english, source_lang="en", target_lang="de")
            aug2_texts = back_translate(aug1_texts, source_lang="en", target_lang="zh")
            
            aug3_texts = back_translate(english, source_lang="en", target_lang="es")
            aug4_texts = back_translate(aug3_texts, source_lang="en", target_lang="fr")
            
            for hi_en, en in zip(hinglish, english):
                data.append([hi_en, en])
            for hi_en, en in zip(hinglish, aug2_texts):
                data.append([hi_en, en])
            for hi_en, en in zip(hinglish, aug4_texts):
                data.append([hi_en, en])
        
        clean_df1 = pd.DataFrame(data, columns=["hi_en", "en"])
        clean_df = clean_df1.drop_duplicates(subset=["en"], keep='first')
        print('clean_df', clean_df.head())
        # clean_df = preprocess(clean_df, 'hi_en')
        # clean_df = preprocess(clean_df, 'en')

        data = []
        for hinglish, english in zip(clean_df['hi_en'], clean_df['en']):
            str2 = str({'prefix':'translate English to Hinglish: ','input': english, 'target': hinglish})
            
            data.append(str2)

        new_df = pd.DataFrame(data, columns=["translation"])
        print(new_df)
        new_df.to_csv('./data/hi_en_bt/cmu_hinglish_dog_{}_mt_en-hi_en.csv'.format(sp))

def CALCS_mt_enghinglish():
    splits = ['train', 'dev', 'test']
    for sp in splits:
        lines = open('../mt_enghinglish/{}.txt'.format(sp), 'r').readlines()
        lines = [l.strip() for l in lines]
        if sp != 'test':
            eng = []
            hi_en = []
            for l in lines:
                eng.append(l.split('\t')[0])
                hi_en.append(l.split('\t')[1])
            
            data = []
            gen = generate_batch(hi_en, eng, 4)
            for hinglish, english in tqdm(gen, total=len(eng)//4+1):
                aug1_texts = back_translate(english, source_lang="en", target_lang="de")
                aug2_texts = back_translate(aug1_texts, source_lang="en", target_lang="zh")
                
                aug3_texts = back_translate(english, source_lang="en", target_lang="es")
                aug4_texts = back_translate(aug3_texts, source_lang="en", target_lang="fr")
                
                for hi_en, en in zip(hinglish, english):
                    data.append([hi_en, en])
                for hi_en, en in zip(hinglish, aug2_texts):
                    data.append([hi_en, en])
                for hi_en, en in zip(hinglish, aug4_texts):
                    data.append([hi_en, en])
            
            clean_df1 = pd.DataFrame(data, columns=["hi_en", "en"])
            clean_df = clean_df1.drop_duplicates(subset=["en"], keep='first')
            print('clean_df', clean_df.head())
            # clean_df = preprocess(clean_df, 'hi_en')
            # clean_df = preprocess(clean_df, 'en')

            data = []
            for hinglish, english in zip(clean_df['hi_en'], clean_df['en']):
                str2 = str({'prefix':'translate English to Hinglish: ','input': english, 'target': hinglish})
                
                data.append(str2)

            new_df = pd.DataFrame(data, columns=["translation"])
            print(new_df)
            new_df.to_csv('./data/hi_en_bt/CALCS_hinglish_{}_mt_en-hi_en.csv'.format(sp))
        else:
            eng = lines
            
            data = []
            for english in eng:
                str2 = str({'prefix':'translate English to Hinglish: ','input': english})
                data.append(str2)
            
            df = pd.DataFrame(data, columns=["translation"])
            print('df', df.head())
            df.to_csv('./data/hi_en_bt/CALCS_hinglish_{}_mt_en-hi_en.csv'.format(sp))
            
def create_mt_data():
    df1 = pd.read_csv('./data/hi_en_bt/cmu_hinglish_dog_train_mt_en-hi_en.csv', index_col=False)
    df2 = pd.read_csv('./data/hi_en_bt/CALCS_hinglish_train_mt_en-hi_en.csv', index_col=False)
    frames = [df1, df2]
    result = pd.concat(frames, ignore_index=True)
    result.drop('Unnamed: 0', axis=1, inplace=True)
    result.to_csv('./data/hi_en_bt/train_mt_en-hi_en.csv')

    df1 = pd.read_csv('./data/hi_en_bt/cmu_hinglish_dog_validation_mt_en-hi_en.csv', index_col=False)
    df2 = pd.read_csv('./data/hi_en_bt/CALCS_hinglish_dev_mt_en-hi_en.csv', index_col=False)
    frames = [df1, df2]
    result = pd.concat(frames, ignore_index=True)
    result.drop('Unnamed: 0', axis=1, inplace=True)
    result.to_csv('./data/hi_en_bt/val_mt_en-hi_en.csv')


def main():
    cmu_hinglish_dog()
    CALCS_mt_enghinglish()
    create_mt_data()



if __name__ == '__main__':
    main()