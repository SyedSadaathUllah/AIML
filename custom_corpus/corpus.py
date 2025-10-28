import nltk
import pandas as pd
import string
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

data = pd.read_csv("custom_corpus.csv")
print(data.head())

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('','',string.punctuation))
    token = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = (word for word in token if word not in stop_words)

    return ' '.join(tokens)
data['cleaned_text'] = data['text'].apply(preprocess)
print(data.head())