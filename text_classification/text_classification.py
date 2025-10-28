import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv("Emotion_classify_Data.csv")
data.dropna(inplace=True)

data.columns = ["text","label"]
print(data.head(2))

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)
data["Clean_text"] = data["text"].apply(preprocess)
print(data[['text','Clean_text']].head())

clean_textvectorizer = TfidfVectorizer()
X = clean_textvectorizer.fit_transform(data['Clean_text'])
y = data['label']
print(X.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("Start",X_train,X_test,y_train,y_test,"End")
model = MultinomialNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("Accuracy :",accuracy_score(y_test,y_pred))
print("Classification Report :",classification_report(y_test,y_pred))

def predict_emotion(text):
    clean = preprocess(text)
    vector = clean_textvectorizer.transform([clean])
    return model.predict(vector)[0]
print(predict_emotion("ive been really angry with r "))