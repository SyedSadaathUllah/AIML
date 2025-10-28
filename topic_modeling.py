import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora,models

data = pd.read_csv('training.1600000.processed.noemoticon.csv',encoding='latin-1',names=['target','ids','date','flag','user','text'])
data = data[['text']].sample(500,random_state=10)
# print(data)

#preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens
data['tokens'] = data['text'].apply(preprocess)
# print(data['tokens'].head(3))


dictionary = corpora.Dictionary(data['tokens'])
corpus = [dictionary.doc2bow(tokens)for tokens in data['tokens']]
# print(corpus[2])

model = models.LdaModel(corpus=corpus,id2word=dictionary,num_topics=5,passes=10,random_state=42)
topics = model.print_topics(num_words=5)
for topic in topics:
    print(topic)

def get_topics(doc_bow):
    topics = model.get_document_topics(doc_bow)
    topics = sorted(topics,key=lambda x:-x[1])
    return topics[0][0] if topics else None

data['topic'] = [get_topics(bow) for bow in corpus]
print(data[['text','topic']].head(10))
