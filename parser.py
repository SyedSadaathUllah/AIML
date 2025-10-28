from nltk.parse.corenlp import CoreNLPParser
from nltk.tokenize import word_tokenize

parser = CoreNLPParser(url='http://localhost:9000')
tokenize = CoreNLPParser(url='http://localhost:9000',tagtype='pos')
pos_tagger = CoreNLPParser(url='http://localhost:9000',tagtype='pos')

sentence = "The quick brown fox jumps over the lazy dog."

token = word_tokenize(sentence)
print("Tokenized sentence:",token)

pos_tag= list(pos_tagger.tag(token))
print("POS tags:",pos_tag)

parser = next(parser.raw_parse(sentence))
print(parser)

parser.pretty_print()