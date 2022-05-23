# QuoteR
Official code and data of the ACL 2022 paper "QuoteR: A Benchmark of Quote Recommendation for Writing"

## 1. Requirements

* nltk==3.5
* numpy==1.19.5
* sklearn==0.0
* torch==1.7.1+cu110
* transformers==3.0.2
* OpenHowNet==0.0.1a11

Note: transformers needs to be installed from source
```
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout b0892fa0e8df02d683e05e625b3903209bff362d
pip install -e .
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I hate you'))"
```

## 2. Usage

###  2.1 Generate sememe data

english_word_sememe.py is used to generate the corresponding language sememe data respectively.

### 2.2  Modify files in the Transformer Python library

Modify the modeling_bert.py file in the Transformer Python library and add three Classes (SememeEmbeddings, BertSememeEmbeddings, BertSememeModel) from bert_english_sememe.py. Then add the BertSememeModel Class to the __init__.py file in the Transformer Python library.

Modify the modeling_distilbert.py file in the Transformer Python library and add three Classes (DistilBertSememeEmbeddings, DistilBertSememeModel) from distilbert_english_sememe.py. Then add the DistilBertSememeModel Class to the __init__.py file in the Transformer Python library.

### 2.3  Model training and testing

The file english_train_test.py is used for training and testing, respectively
