# Description
This code is for "Question difficulty prediction for multiple choice problems in medical exams".


# Set Environments
We first need to create a *python=3.6* virtualenv and activate it.

Then, we should intall some dependencies.
```shell
pip install -r requirements.txt
``` 


# Prepare Data
The corpus consists of three files *train.txt*, *dev.txt*, and *test.txt*, which are put in *data* folder.
The format of each line in these files is 

```json
{
    "id": XXX,
    "difficulty": 0.XXX,
    "question": "TEXT OF QUESTION",
    "answer": "B",
    "A": "OPTION A", "B": "OPTION B", "C": "OPTION C", "D": "OPTION D", "E": "OPTION E",
    "es_research_facts": {
        "Q+A": ["XXXX", "XXXX", ...],
        "Q+B": ["XXXX", "XXXX", ...],
        "Q+C": ["XXXX", "XXXX", ...],
        "Q+D": ["XXXX", "XXXX", ...],
        "Q+E": ["XXXX", "XXXX", ...],
    }
}
```

## Pre-process
Build corpus

```shell
cat data/train.txt data/dev.txt data/test.txt > data/total.txt
python -m scripts.build_corpus --f data/total.txt
```

Build vocab

```shell
python -m scripts.build_vocab -c data/total.txt -s 40000 -wo data/word_embeddings.txt
```

## Construct dataset
```shell
python -m scripts.build_dataset --f data/train.txt -e 10 --o data/train_set.txt
python -m scripts.build_dataset --f data/dev.txt -e 10 --o data/dev_set.txt
python -m scripts.build_dataset --f data/test.txt -e 10 --o data/test_set.txt
```

# Training and Testing
```shell
# Training
python main.py -cf conf.ini --mode 0 
```

```shell
# Testing
python main.py -cf conf.ini --mode 1 --epoch_for_test 1
```

# Citation
If you uses this code, please cite the paper.
```
@inproceedings{DBLP:conf/cikm/QiuW019,
  author    = {Zhaopeng Qiu and
               Xian Wu and
               Wei Fan},
  title     = {Question Difficulty Prediction for Multiple Choice Problems in Medical Exams},
  booktitle = {{CIKM}},
  pages     = {139--148},
  publisher = {{ACM}},
  year      = {2019}
}
```
