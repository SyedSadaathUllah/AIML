import nltk
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.tokenize import word_tokenize
from nltk import pos_tag,ne_chunk
import pandas as pd

data = pd.read_csv("training.1600000.processed.noemoticon.csv",
                   encoding='latin-1',
                   names=['target','ids','date','flag','user','text'])
# print(data.head(5))
#
# tweet = data['text'][1]
# print(f"Tweet : {tweet}")
#
# tokens = word_tokenize(tweet)
# postag = pos_tag(tokens)
# print(tokens)
# print(postag)
#
# sense = lesk(tokens,'School')
# print(sense)
# print(sense.definition())
#
# dog = wn.synsets('good', pos=wn.ADJ)[0]
# bad = wn.synsets('bad', pos=wn.ADJ)[0]
# similarity = dog.wup_similarity(bad)
# print(f"Semantic similarity : {similarity}")
#
# tree = ne_chunk(postag)
# print(tree)
# tree.pretty_print()

def semantic_analysis(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    nechunk = ne_chunk(tags)
    print("Original",text)
    print("Tokenized",tokens)
    print("tagged",tags)
    print("ne_chunks",nechunk)

    for word in tokens:
        synsets = wn.synsets(word)
        if synsets:
            print(f"\n Word:{word}")
            for syn in synsets:
                print(f" - {syn.name()}:{syn.definition()}")
    print("\n")
for i in range(2):
    semantic_analysis(data['text'][i])