import spacy
import pandas as pd
from nltk.corpus import names
from torch.distributions.constraints import dependent

data = pd.read_csv('training.1600000.processed.noemoticon.csv',
                   encoding='latin-1',
                   names=['target', 'ids', 'date', 'flag', 'user', 'text'])
data = data[['text']]
print(data.head(3))

nlp = spacy.load("en_core_web_sm")
for i, row in data.head(5).iterrows():
    doc = nlp(row['text'])
    print(f"\n Tweet :{row['text']}")
    for token in doc:
        print(f"{token.text:<12} | Head : {token.head.text:<12} | Dep :{token.dep_:<12} | POS:{token.pos_}")

parsed_rows = []
for i, row in data.head(5).iterrows():
    doc = nlp(row['text'])
    deps = [(token.text,token.dep_,token.head.text) for token in doc]
    parsed_rows.append({'text':row['text'],'dependencies':deps})
parsed_data = pd.DataFrame(parsed_rows)
print(parsed_data.head())