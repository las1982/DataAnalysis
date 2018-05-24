# import nltk, sys
# nltk.download("wordnet")
# sys.exit(0)

from nltk.stem.api import StemmerI
from nltk.stem.regexp import RegexpStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.isri import ISRIStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.rslp import RSLPStemmer

from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
print(wordnet_lemmatizer.lemmatize("dogs"))
print(wordnet_lemmatizer.lemmatize("maximum"))

stemmers = [
    PorterStemmer()
    ,LancasterStemmer()
    ,SnowballStemmer("english")
]
lemmatizers = [
    WordNetLemmatizer()
]
for stemmer in stemmers:
    print(stemmer.stem("maximum"))
