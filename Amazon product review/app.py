import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
import tensorflow as tf
import warnings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.layers import SimpleRNN,LSTM,Dense,Dropout,Embedding,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
warnings.filterwarnings('ignore')

data = pd.read_csv('AmazonReview.csv')
# print(data[:100])
# print(data.info())

# print("Null Values : ",data.isna().sum())
data = data.dropna()
# print(data.isna().sum())
count = data['Sentiment'].value_counts()
# print(count)

stop_words = set(stopwords.words('english'))

def clean_reviews(text):
    regex = re.compile('<,*?>')
    text = re.sub(regex,' ',text)

    pattern = re.compile('[^a-zA-Z0-9\s]')
    text =  re.sub(pattern, ' ',text)

    pattern = re.compile('\d+')
    text = re.sub(pattern,' ',text)

    text = text.lower()

    text = word_tokenize(text)

    text = [word for word in text if not word in stop_words]

    return text
data['Review']=data['Review'].apply(clean_reviews)

tokenizer = Tokenizer()
reviews_to_list = data['Review'].tolist()
tokenizer.fit_on_texts(reviews_to_list)
text_sequence = tokenizer.texts_to_sequences(reviews_to_list)
print(text_sequence)