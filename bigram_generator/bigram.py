import nltk
from nltk.tokenize import word_tokenize
from nltk.util import bigrams

text = "Vinay is an Indian name typically meaning guidance, good behaviour, genuinity, politeness, modesty and smart in Sanskrit. It has its origins in the Sanskrit language origin. "

token = word_tokenize(text)

bigrams_list = list(bigrams(token))

for bi_grams in bigrams_list:
    print(bi_grams)
