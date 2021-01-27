import pandas as pd
import numpy as np
import nltk
import pickle
import argparse
from operator import itemgetter

from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from nltk import tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

# referate_train = pd.read_json('referate-train.json')
referate_dev = pd.read_json("referate-test.json")
# referate_test = pd.read_json('referate-test.json')

model = Doc2Vec.load("doc2vec.model")

stop = stopwords.words("romanian")
stop_words = set(stopwords.words("romanian"))
wn = WordNetLemmatizer()


def black_txt(token):
    return (
        token not in stop_words
        and token not in list(string.punctuation)
        and len(token) > 2
    )


def clean_txt(text):
    clean_text = []
    clean_text2 = []
    text = re.sub("'", "", text)
    text = re.sub("(\\d|\\W)+", " ", text)
    text = text.replace("nbsp", "")
    clean_text = [
        wn.lemmatize(word, pos="v")
        for word in word_tokenize(text.lower())
        if black_txt(word)
    ]
    clean_text2 = [word for word in clean_text if black_txt(word)]
    return " ".join(clean_text2)


def tokenize(text):
    return word_tokenize(clean_txt(text).lower())


# referate_train['text'] = referate['text'].apply(clean_txt)
referate_dev["text_tokenized"] = referate_dev["text"].apply(tokenize)
print("Preprocessing done !")

embeddings = [model.infer_vector(x) for x in referate_dev["text_tokenized"].values]
print("All embeddings done !")

referate_dev["embeddings"] = embeddings

referate_dev.to_json("referate-test-embeddings.json")
print("Dataframe saved !")
