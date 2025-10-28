from probability import word_tag_model
from nltk.corpus import treebank
from nltk.tag import UnigramTagger

train_data = treebank.tagged_sents()[:30]
test_data = treebank.tagged_sents()[30:]
# print(train_data)
# print(test_data)

model = word_tag_model(treebank.words(),
                       treebank.tagged_words())
tag = UnigramTagger(model=model)
print("Accuracy :",tag.evaluate(test_data))
