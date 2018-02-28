import dill as pickle
# import pickle
import jieba
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import codecs

from profitero_data_scientist.utils.Constants import File, Vectorizers


class Vectorizer:

    def __init__(self, vectorizer_name):
        self.vectorizer = None
        self.__vectorizer_name__ = vectorizer_name
        self.__vectorizer_file__ = '/'.join([File.vectorizers_dir, self.__vectorizer_name__])

        if vectorizer_name == Vectorizers.ch_revi_1_5_ngram_count:
            self.vectorizer = CountVectorizer(
                input='content',
                encoding='utf-8',
                decode_error='strict',
                strip_accents=None,
                lowercase=False,
                preprocessor=None,
                tokenizer=lambda text: jieba.cut(text, cut_all=False, HMM=False),
                stop_words=[line.strip() for line in codecs.open(File.chinese_stop_words, "r", encoding="utf-8")],
                # token_pattern=r"(?u)\b\w\w+\b",
                ngram_range=(1, 5),
                analyzer='word',
                max_df=0.7,
                min_df=0.002,
                max_features=None,
                vocabulary=None,
                binary=False,
                dtype=np.int64
            )

        if vectorizer_name == Vectorizers.ch_prod_1_2_ngram_count:
            self.vectorizer = CountVectorizer(
                input='content',
                encoding='utf-8',
                decode_error='strict',
                strip_accents=None,
                lowercase=False,
                preprocessor=None,
                tokenizer=lambda text: jieba.cut(text, cut_all=False, HMM=False),
                stop_words=[line.strip() for line in codecs.open(File.chinese_stop_words, "r", encoding="utf-8")],
                # token_pattern=r"(?u)\b\w\w+\b",
                ngram_range=(1, 2),
                analyzer='word',
                max_df=0.7,
                min_df=0.002,
                max_features=None,
                vocabulary=None,
                binary=False,
                dtype=np.int64
            )

        if vectorizer_name == Vectorizers.en_revi_1_5_ngram_count:
            self.vectorizer = CountVectorizer(
                input='content',
                encoding='utf-8',
                decode_error='strict',
                strip_accents=None,
                lowercase=True,
                preprocessor=None,
                tokenizer=None,
                stop_words='english',
                token_pattern=r"(?u)\b\w\w+\b",
                ngram_range=(1, 5),
                analyzer='word',
                max_df=0.7,
                min_df=0.002,
                max_features=None,
                vocabulary=None,
                binary=False,
                dtype=np.int64
            )

        if vectorizer_name == Vectorizers.en_prod_1_2_ngram_count:
            self.vectorizer = CountVectorizer(
                input='content',
                encoding='utf-8',
                decode_error='strict',
                strip_accents=None,
                lowercase=True,
                preprocessor=None,
                tokenizer=None,
                stop_words='english',
                token_pattern=r"(?u)\b\w\w+\b",
                ngram_range=(1, 2),
                analyzer='word',
                max_df=0.7,
                min_df=0.002,
                max_features=None,
                vocabulary=None,
                binary=False,
                dtype=np.int64
            )

    def fit(self, X):
        self.vectorizer.fit(X)
        self.__save__()

    def __save__(self):
        pickle.dump(self.vectorizer, open(self.__vectorizer_file__, 'wb'))

    def __load__(self):
        self.vectorizer = pickle.load(open(self.__vectorizer_file__, 'rb'))

    def transform(self, X):
        self.__load__()
        return self.vectorizer.transform(X)
