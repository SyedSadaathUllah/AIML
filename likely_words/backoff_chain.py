from nltk import BigramTagger, TrigramTagger
from nltk.tag import UnigramTagger
from nltk.tag import DefaultTagger
from likely_words.unigram import train_data, test_data
from unigram import model

default_tagger = DefaultTagger('NN')

likely_tagger = UnigramTagger(model=model, backoff=default_tagger)


def backoff_tagger(train_data, tagger_classes, backoff=None):
    tagger = backoff
    for cls in tagger_classes:
        tagger = cls(train_data, backoff=tagger)
    return tagger


tag = backoff_tagger(train_data,
                     [UnigramTagger, BigramTagger, TrigramTagger],
                     backoff=likely_tagger)
print("Accuracyyy :", tag.evaluate(test_data))
