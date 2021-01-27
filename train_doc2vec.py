import pandas as pd
import numpy as np
import nltk
import pickle

from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from nltk import tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize

stop = stopwords.words('romanian')
stop_words = set(stopwords.words('romanian'))
wn = WordNetLemmatizer()

def black_txt(token):
    return token not in stop_words and token not in list(string.punctuation) and len(token)>2

def clean_txt(text):
    clean_text = []
    clean_text2 = []
    text = re.sub("'", "",text)
    text=re.sub("(\\d|\\W)+"," ",text) 
    text = text.replace("nbsp", "")
    clean_text = [ wn.lemmatize(word, pos="v") for word in word_tokenize(text.lower()) if black_txt(word)]
    clean_text2 = [word for word in clean_text if black_txt(word)]
    return " ".join(clean_text2)

referate = pd.read_json('referate-train.json')
referate['text'] = referate['text'].apply(clean_txt)

tokenized_doc = []
for d in referate['text']:
    tokenized_doc.append(word_tokenize(d.lower()))

tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]

#tagged_data = [TaggedDocument(d, i) for i, d in enumerate(zip(tokenized_doc, ai['id'].values))]
#print(tagged_data)

model = Doc2Vec(tagged_data, vector_size = 200, window = 10, min_count = 1, workers = 4, epochs = 100)

model.save("doc2vec_200.model")