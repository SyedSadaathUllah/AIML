import nltk
import random
from nltk.corpus import names

def gender_features(word):
    return {'last letter' : word[-1]}

labeled_names = ([(name,'male')for name in names.words('male.txt')]+
                 [(name,'female')for name in names.words('female.txt')])

random.shuffle(labeled_names)
# print(labeled_names[:10])
featuresets = [(gender_features(n),gender) for (n,gender) in labeled_names]

train_sets,test_sets = featuresets[500:],featuresets[:500]
# print(train_sets[:10])
# print(test_sets[:10])

classifier = nltk.NaiveBayesClassifier.train(train_sets)
print(classifier.classify(gender_features(input("Enter Your name : "))))

print(nltk.classify.accuracy(classifier,train_sets))

# classifier.show_most_informative_features(10)
