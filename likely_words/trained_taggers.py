from nltk import BigramTagger, TrigramTagger
from nltk.tag import UnigramTagger
from nltk.tag import DefaultTagger
from likely_words.backoff_chain import backoff_tagger
from likely_words.unigram import model
from unigram import train_data,test_data
default_tagger = DefaultTagger('NN')

tagger = backoff_tagger(train_data,[UnigramTagger,BigramTagger,TrigramTagger],backoff=default_tagger)
likely_tag = UnigramTagger(model=model,backoff=tagger)
print("Accuracy : ",likely_tag.evaluate(test_data))