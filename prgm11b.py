import nltk
# nltk.download()
from nltk.util import ngrams
x = 'welcome to amal jyothi college of engineering'
y = ngrams(sequence=nltk.word_tokenize(x), n=3)
for i in y:
 print(i)