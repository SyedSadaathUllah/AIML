from nltk.probability import FreqDist,ConditionalFreqDist

def word_tag_model(words,tagged_words,limit = 200):
    fd = FreqDist(words)
    cfd = ConditionalFreqDist(tagged_words)
    most_freq = (word for word,count in fd.most_common(limit))

    return dict((word,cfd[word].max())
                for word in most_freq)
