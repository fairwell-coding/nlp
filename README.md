# Neural Machine Translation For English-Hinglish
```
Sayan Mandal, Stefan Schörkmeier, Philipp Temmel
Technische Universität Graz
June, 2022
```

We train NMT models based on t5-small for English->Hinglish and Hinglish->English

All models trained on AMD Ryzen 7 3800XT, GTX 1070 and available on [HuggingFace](https://huggingface.co/sayanmandal)

For LinCE benchmark on English->Hinglish, check this [leaderboard](https://ritual.uh.edu/lince/leaderboard)

## Guide
```
- sudo apt-get install python3-pip
- pip install virtualenv
- virtualenv -p /usr/bin/python3.8 nmt
- git clone https://github.com/fairwell-coding/nlp.git
- cd nlp
- pip install -r requirements.txt
- python pipeline.py
```
- Have a look at the argument list in pipeline.py

## Student Model
We create a student model from t5-small but we do not use distillation afterwards. Only fine-tuning.
To create student model run:
```
python make_student.py t5-small t5_small_6_3 6 3
```

## Data Creation
We use two datasets in this work:
- [cmu_hinglish_dog](https://huggingface.co/datasets/cmu_hinglish_dog)
- [LinCE English-Hinglish](https://ritual.uh.edu/lince/datasets)

We train them jointly and evaluate seperately.

For Hinglish-English:
1. Run:
```
python preprocess_data_en.py
```
2. Train split available [here](./data/en/train_mt_hi_en-en.csv) and val split available [here](./data/en/val_mt_hi_en-en.csv)
3. [LinCE English-Hinglish](https://ritual.uh.edu/lince/datasets) does not have test Hinglish texts so testing can only be done on [cmu_hinglish_dog](https://huggingface.co/datasets/cmu_hinglish_dog), test split available [here](./data/en/cmu_hinglish_dog_test_mt_hi_en-en.csv)

For English-Hinglish (w/o Back Translation):
1. Run:
```
python preprocess_data_hien.py
```
2. Train split available [here](./data/hi_en/train_mt_en-hi_en.csv) and val split available [here](./data/hi_en/val_mt_en-hi_en.csv)
3. For [LinCE English-Hinglish](https://ritual.uh.edu/lince/datasets), test split available [here](./data/hi_en/CALCS_hinglish_test_mt_en-hi_en.csv) and for [cmu_hinglish_dog](https://huggingface.co/datasets/cmu_hinglish_dog), test split available [here](./data/hi_en/cmu_hinglish_dog_test_mt_en-hi_en.csv)

For English-Hinglish (with Back Translation augmentations):
1. Run:
```
python preprocess_data_hien_bt.py
```
2. Train split available [here](./data/hi_en_bt/cmu_hinglish_dog_train_mt_en-hi_en.csv) and val split available [here](./data/hi_en_bt/cmu_hinglish_dog_validation_mt_en-hi_en.csv)
3. For [LinCE English-Hinglish](https://ritual.uh.edu/lince/datasets), test split available [here](./data/hi_en/CALCS_hinglish_test_mt_en-hi_en.csv) and for [cmu_hinglish_dog](https://huggingface.co/datasets/cmu_hinglish_dog), test split available [here](./data/hi_en/cmu_hinglish_dog_test_mt_en-hi_en.csv) (same as w/o augmentation, ofcourse :P)

## Model Training

To login into your HuggingFace account and share model, run:
```
huggingface-cli login
```

If you don't want to share model, run without ```--push_to_hub```

Example: Hinglish->English

Run:
```
python pipeline.py --train True --push_to_hub True --prefix "translate English to Hinglish: " --source_lang "en" --target_lang "hi_en" --train_path "./data/hi_en/train_mt_en-hi_en.csv" --val_path "./data/hi_en/val_mt_en-hi_en.csv" --test_path "./data/hi_en/cmu_hinglish_dog_test_mt_en-hi_en.csv"
```

Change paths and hyperparameters as per necessity. Have a look at pipline.py line 266-294 

## Model Testing

### Hinglish->English

Weight: sayanmandal/t5-small_6_3-hi_en-en_mix ([link](https://huggingface.co/sayanmandal/t5-small_6_3-hi_en-en_mix))

Dataset: [cmu_hinglish_dog](https://huggingface.co/datasets/cmu_hinglish_dog) + [LinCE](https://ritual.uh.edu/lince/datasets)

Run:
```
python pipeline.py --prefix "translate Hinglish to English: " --source_lang "hi_en" --target_lang "en" --model_checkpoint "sayanmandal/t5-small_6_3-hi_en-en_mix" --tokenizer_checkpoint "sayanmandal/t5-small_6_3-hi_en-en_mix"
```

**Results:**

cmu_hinglish_dog val BLEU: **18.7011**, test BLEU: **19.4963**

### English->Hinglish (trained on LinCE w/o back translation)

Weight: sayanmandal/t5-small_6_3-en-hi_en_LinCE ([link](https://huggingface.co/sayanmandal/t5-small_6_3-en-hi_en_LinCE))

Dataset: [LinCE](https://ritual.uh.edu/lince/datasets)

Run:
```
python pipeline.py --prefix "translate English to Hinglish: " --source_lang "en" --target_lang "hi_en" --model_checkpoint "sayanmandal/t5-small_6_3-en-hi_en_LinCE" --tokenizer_checkpoint "sayanmandal/t5-small_6_3-en-hi_en_LinCE" --test_save_path "./mt_eng_hinglish.txt"
```

**Results:**

cmu_hinglish_dog val BLEU: **8.1099**, test BLEU: **7.8671**, LinCE test BLEU: **4.88**

### English->Hinglish (trained on LinCE with back translation)

Weight: sayanmandal/t5-small_6_3-en-hi_en_LinCE_bt ([link](https://huggingface.co/sayanmandal/t5-small_6_3-en-hi_en_LinCE_bt))

Dataset: [LinCE](https://ritual.uh.edu/lince/datasets)

Run:
```
python pipeline.py --prefix "translate English to Hinglish: " --source_lang "en" --target_lang "hi_en" --model_checkpoint "sayanmandal/t5-small_6_3-en-hi_en_LinCE_bt" --tokenizer_checkpoint "sayanmandal/t5-small_6_3-en-hi_en_LinCE_bt" --test_save_path "./mt_eng_hinglish.txt"
```

**Results:**

cmu_hinglish_dog val BLEU: **9.5527**, test BLEU: **9.4473**, LinCE test BLEU: **4.97**

### English->Hinglish (w/o back translation)

Weight: sayanmandal/t5-small_6_3-en-hi_en__noBT ([link](https://huggingface.co/sayanmandal/t5-small_6_3-en-hi_en__noBT))

Dataset: [cmu_hinglish_dog](https://huggingface.co/datasets/cmu_hinglish_dog) + [LinCE](https://ritual.uh.edu/lince/datasets)

Run:
```
python pipeline.py --prefix "translate English to Hinglish: " --source_lang "en" --target_lang "hi_en" --model_checkpoint "sayanmandal/t5-small_6_3-en-hi_en__noBT" --tokenizer_checkpoint "sayanmandal/t5-small_6_3-en-hi_en__noBT" --test_save_path "./mt_eng_hinglish.txt"
```

**Results:**

cmu_hinglish_dog val BLEU: **9.3829**, test BLEU: **9.8813**, LinCE test BLEU: **5.47**

### English->Hinglish (with back translation)

Weight: sayanmandal/t5-small_6_3-en-hi_en_bt ([link](https://huggingface.co/sayanmandal/t5-small_6_3-en-hi_en_bt))

Dataset: [cmu_hinglish_dog](https://huggingface.co/datasets/cmu_hinglish_dog) + [LinCE](https://ritual.uh.edu/lince/datasets)

Run:
```
python pipeline.py --prefix "translate English to Hinglish: " --source_lang "en" --target_lang "hi_en" --model_checkpoint "sayanmandal/t5-small_6_3-en-hi_en_bt" --tokenizer_checkpoint "sayanmandal/t5-small_6_3-en-hi_en_bt" --test_save_path "./mt_eng_hinglish.txt"
```

**Results:**

cmu_hinglish_dog val BLEU: **9.2397**, test BLEU: **9.1481**, LinCE test BLEU: **4.71**

## Inference

For Hinglish->English, run:
```
python inference.py --prefix "translate Hinglish to English: " --model_checkpoint "sayanmandal/t5-small_6_3-hi_en-en_mix" --tokenizer_checkpoint "sayanmandal/t5-small_6_3-hi_en-en_mix" --input "Ye movie kis baare main hai?"
```

Change path of csv in line 97 and 107 of inference.py

```
Source: "Ye movie kis baare main hai?"
Target: "What is the movie about?"
Predicted: "What is the movie about?"
```

## Distillation 
To execute the distillation execute the following steps:
1. download the transformers repository from `https://github.com/huggingface/transformers` and paste it in the root folder.
2. install the requirements.txt in the root folder.
3. install the requirements.txt in `./transformers-main/examples/research_projects/seq2seq-distillation`.
4. execute the `distillation_script.sh`, change parameters if needed (e.g. number ob decoder / encoder).

**Results:**

Distillation with 2 encoders and 1 decoders: 1.8104. \
Distillation with 6 encoders and 3 decoders: 2.0647.

## Progress
- [x] Dataset Creation
- [x] Student Model Creation
- [x] Model Finetune
- [x] Model Test
- [x] Model Inference
- [x] Exps ...
- [x] Publish to HuggingFace
- [x] Submit to LinCE [leaderboard](https://ritual.uh.edu/lince/leaderboard)

## Citation
```
@article{raffel2020exploring,
  title={Exploring the limits of transfer learning with a unified text-to-text transformer.},
  author={Raffel, Colin and Shazeer, Noam and Roberts, Adam and Lee, Katherine and Narang, Sharan and Matena, Michael and Zhou, Yanqi and Li, Wei and Liu, Peter J and others},
  journal={J. Mach. Learn. Res.},
  volume={21},
  number={140},
  pages={1--67},
  year={2020}
}

@inproceedings{aguilar-etal-2020-lince,
                    title = "{L}in{CE}: {A} {C}entralized {B}enchmark for {L}inguistic {C}ode-switching {E}valuation",
                    author = "Aguilar, Gustavo  and
                    Kar, Sudipta  and
                    Solorio, Thamar",
                    booktitle = "Proceedings of The 12th Language Resources and Evaluation Conference",
                    month = may,
                    year = "2020",
                    address = "Marseille, France",
                    publisher = "European Language Resources Association",
                    url = "https://www.aclweb.org/anthology/2020.lrec-1.223",
                    pages = "1803--1813",
                    language = "English",
                    ISBN = "979-10-95546-34-4",
}

@inproceedings{
    cmu_dog_emnlp18,
    title={A Dataset for Document Grounded Conversations},
    author={Zhou, Kangyan and Prabhumoye, Shrimai and Black, Alan W},
    year={2018},
    booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing}
}

@inproceedings{post-2018-call,
  title = "A Call for Clarity in Reporting {BLEU} Scores",
  author = "Post, Matt",
  booktitle = "Proceedings of the Third Conference on Machine Translation: Research Papers",
  month = oct,
  year = "2018",
  address = "Belgium, Brussels",
  publisher = "Association for Computational Linguistics",
  url = "https://www.aclweb.org/anthology/W18-6319",
  pages = "186--191",
}
```
