# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:38:41 2019

@author: YQ
"""

from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from itertools import chain

import re
import pickle
import pandas as pd

def preprocess_sentence(s):
    s = s.lower()
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)
    s = s.rstrip().strip()
    
    return s

df = pd.read_csv("data/imdb_master.csv", encoding="latin-1")
sentences = df["review"].values.tolist()

# split string by sentence
sentences = [re.sub("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", "\n", s) for s in sentences]
sentences = [s.split("\n") for s in sentences]

# OPTIONAL: keep only sentences with less than 200 words
sentences = list(chain(*sentences))
sentences = [preprocess_sentence(s) for s in sentences]
sentences = [s for s in sentences if len(s) <= 200]

xtr, xte = train_test_split(sentences, test_size=100000)

# OPTIONAL: keep vocabulary size at 20000 words + 1 token for unknown <unk> token
tokenizer = Tokenizer(20000+1, filters="", lower=False, oov_token="<unk>")
tokenizer.fit_on_texts(xtr)

xtr = tokenizer.texts_to_sequences(xtr)
xte = tokenizer.texts_to_sequences(xte)

# append a 0 to indicate end of sentence
xtr = [s + [0] for s in xtr]
xte = [s + [0] for s in xte]

pickle.dump(xtr, open("data/imdb_train.p", "wb"))
pickle.dump(xte, open("data/imdb_valid.p", "wb"))
pickle.dump(tokenizer, open("data/imdb_tokenizer.p", "wb"))

