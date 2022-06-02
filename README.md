# Neural Machine Translation

## Guide
```
- sudo apt-get install python3-pip
- pip install virtualenv
- virtualenv -p /usr/bin/python3.8 nmt
- git clone https://github.com/fairwell-coding/nlp.git
- cd nlp
- pip install -r requirements.txt
- python boilerplate.py
```
- Have a look at the argument list in boilerplate.py

### Distillation
To execute the distillation exectue the following steps:
1. install the requirements.txt in the root folder. 
2. install the requirements.txt in `./distillation/transformers-main/examples/research_projects/seq2seq-distillation`.
3. execute either `distillation_script.sh` or `make_student_script.sh`

## Progress
- [x] Boilerplate Finetune
- [ ] Boilerplate Test
- [ ] Boilerplate Inference
- [ ] Exps ...
- [ ] Publish to HuggingFace
- [ ] Visualization maybe? (https://github.com/jessevig/bertviz)

## Issues
- WMT'14, WMT'19 too big for GTX1070. Discuss ...

## Note
- Do NOT push weights, dataset here!