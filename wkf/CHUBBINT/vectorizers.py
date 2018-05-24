import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


class NewCountVectorizer(CountVectorizer):
    lemmatizer = WordNetLemmatizer()

    # stemmer = SnowballStemmer("english")
    # analyzer = CountVectorizer().build_analyzer()

    def __init__(self,
                 input='content',
                 encoding='utf-8',
                 decode_error='strict',
                 strip_accents=None,
                 lowercase=True,
                 preprocessor=lambda text: re.sub("\d+|'| a-z ", " ", text.lower()),
                 # preprocessor=None,
                 tokenizer=None,
                 stop_words=set(stopwords.words('english')).union({"ee", "eployee", "employee", "ee's"}),
                 # stop_words=None,
                 token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1),
                 analyzer='word',
                 max_df=1.0,
                 min_df=1,
                 max_features=None,
                 vocabulary=None,
                 binary=False,
                 dtype=np.int64):
        super().__init__(input, encoding, decode_error, strip_accents, lowercase, preprocessor, tokenizer, stop_words,
                         token_pattern, ngram_range, analyzer, max_df, min_df, max_features, vocabulary, binary, dtype)
        self.stop = stop_words

    # def build_tokenizer(self):
    #     self.tokenizer = super(NewCountVectorizer, self).build_tokenizer()
    #     return lambda doc: self.rrr(doc)
    #     # return lambda doc: (self.lemmatizer.lemmatize(w) for w in self.tokenizer(doc) if w not in self.stop)

    def rrr(self, doc):
        words = [self.lemmatizer.lemmatize(w) for w in self.tokenizer(doc) if w not in self.stop]
        # words = [stemmer.stem(lemmatizer.lemmatize(w)) for w in tokenizer(doc) if w not in self.stop]
        # return ([lemmatizer.lemmatize(w) for w in analyzer(doc) if w.lower() not in self.stop])
        return ([self.lemmatizer.lemmatize(w) for w in self.tokenizer(doc) if w not in self.stop])
        # return ([stemmer.stem(lemmatizer.lemmatize(w)) for w in tokenizer(doc) if w not in self.stop])

    def prepared_words(self, doc):
        words = (self.lemmatizer.lemmatize(w) for w in self.tokenizer(doc) if w not in self.stop)
        return words


# vect = NewCountVectorizer(
#         input='content',
#         encoding='utf-8',
#         decode_error='strict',
#         strip_accents=None,
#         lowercase=True,
#         preprocessor=lambda doc: re.sub("\d+", "", doc.lower()),
#         tokenizer=None,
#         token_pattern=r"(?u)\b\w\w+\b",
#         ngram_range=(1, 4),
#         analyzer='word',
#         max_df=1.0,
#         min_df=1,
#         max_features=None,
#         vocabulary=None,
#         binary=False,
#         dtype=np.int64
#     )
# vect.fit(["EE sustained a soft tissue injury to her left"])