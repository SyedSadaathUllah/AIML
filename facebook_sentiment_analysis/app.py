import time
from traceback import print_tb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import os
import unicodedata
import re
import string
from numpy import linalg
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import webtext
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tag import pos_tag

with open('fb_sentiment.csv',encoding='ISO-8859-2') as f:
    text = f.read()
# print(text)
# sent_tokenizer = sent_tokenize(text)
# word_tokenizer = word_tokenize(text)
# # print(sent_tokenizer)
# # print(word_tokenizer)
#
# port_stemmer = PorterStemmer()
#
# for w in word_tokenizer:
#     stemmer = (w,port_stemmer.stem(w))
#     # print("Actual Word : %s Stemmed word : %s" % (w,port_stemmer.stem(w)))
#     # print(stemmer)
#
# lemmatizer = WordNetLemmatizer()
#
# for w in word_tokenizer:
#     lemmatized = (w, lemmatizer.lemmatize(w))
#     # print("Actual Word : % s Lemmatized word : %s" % (w, lemmatizer.lemmatize(w)))
#     # print(lemmatized)
#
# pos_tag = pos_tag(word_tokenizer)
# # print(pos_tag)

sid = SentimentIntensityAnalyzer()

with open('fb_sentiment.csv',encoding='ISO-8859-2')as f:
    for text in f.read().split('\n'):
        print(text)
        scores = sid.polarity_scores(text)
        for key in sorted(scores):
            print('{0}:{1},'.format(key,scores[key]),end='')
    print()