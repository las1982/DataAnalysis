import pickle
from scipy import sparse

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
# from nltk.parse.corenlp import CoreNLPTokenizer
import os
import sys

from datetime import datetime

# df = pd.read_table('jd_reviews.csv', encoding='utf-8', quotechar='"', sep=',', dtype='str', na_values=["nan"], keep_default_na=False)
from profitero_data_scientist.utils.Enums import Field
from profitero_data_scientist.utils.Enums import File
import jieba
import codecs

df_jd_reviews = pd.read_table(File.jd_reviews, encoding='utf-8', quotechar='"', sep=',', dtype='str')
df_jd_reviews.dropna(inplace=True)

df_review_en = pd.read_table(File.review_en, encoding='utf-8', quotechar='"', sep=',', dtype='str',
                             usecols=[Field.review, Field.review_en])
df_product_en = pd.read_table(File.product_en, encoding='utf-8', quotechar='"', sep=',', dtype='str',
                              usecols=[Field.product_name, Field.product_name_en])

df = df_jd_reviews.merge(df_review_en, left_on=Field.review, right_on=Field.review, how='left')
df = df.merge(df_product_en, left_on=Field.product_name, right_on=Field.product_name, how='left')
df = df[[
    Field.review_id,
    Field.created_at,
    Field.review,
    Field.review_en,
    Field.product_id,
    Field.product_name,
    Field.product_name_en,
    Field.stars
]]

# df = df.sample(frac=0.01, random_state=10)


def tokenize_and_delete_stop_words(old_text, stopwords):
    new_text = []
    for word in jieba.cut(old_text, cut_all=False):
        if word not in stopwords and word not in [' ', '\n', '\r']:
            new_text.append(word)
    return ' '.join(new_text)


stopwords = [line.rstrip() for line in
             codecs.open('/home/alex/work/projects/DataAnalysis/profitero_data_scientist/chinese_stop_words.txt',
                         "r", encoding="utf-8")]

df[Field.created_at_day_of_week] = df[Field.created_at].apply(
    lambda date: datetime.strptime(date, '%Y-%m-%d').weekday())
df[Field.review_tokenized] = df[Field.review].apply(
    lambda old_text: tokenize_and_delete_stop_words(old_text, stopwords))
df[Field.product_name_tokenized] = df[Field.product_name].apply(
    lambda old_text: tokenize_and_delete_stop_words(old_text, stopwords))

# print(df.head())
# sys.exit(0)

# predictor = Field.review_tokenized
# X = df[[predictor]]

X = df[[Field.review_tokenized, Field.created_at_day_of_week, Field.product_name_tokenized]]
y = df[[Field.stars]]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# vect = CountVectorizer().fit(X_train[Field.review_en])
# vect = CountVectorizer(min_df=3, ngram_range=(3, 4)).fit(X_train[predictor])
vect_review = CountVectorizer(ngram_range=(1, 5)).fit(X_train[Field.review_tokenized])
vect_product = CountVectorizer(ngram_range=(1, 3)).fit(X_train[Field.product_name_tokenized])


# vect = TfidfVectorizer(min_df=5).fit(X_train[predictor])


def sprs_matr_to_df(matr):
    matr_to_df = pd.DataFrame()
    for i in range(matr.shape[0]):
        matr_to_df = matr_to_df.append(pd.DataFrame(matr[i, :].toarray()))
    matr_to_df.reset_index(inplace=True)
    return matr_to_df


X_train_vectorized = sprs_matr_to_df(vect_review.transform(X_train[Field.review_tokenized]))
tmp_df = sprs_matr_to_df(vect_product.transform(X_train[Field.product_name_tokenized]))
X_train_vectorized = X_train_vectorized.merge(tmp_df, left_index=True, right_index=True, how='left')
X_train_vectorized[Field.created_at_day_of_week] = X_train[Field.created_at_day_of_week].values

X_test_vectorized = sprs_matr_to_df(vect_review.transform(X_test[Field.review_tokenized]))
tmp_df = sprs_matr_to_df(vect_product.transform(X_test[Field.product_name_tokenized]))
X_test_vectorized = X_test_vectorized.merge(tmp_df, left_index=True, right_index=True, how='left')
X_test_vectorized[Field.created_at_day_of_week] = X_test[Field.created_at_day_of_week].values

# X_test_vectorized = vect_review.transform(X_test[Field.review_tokenized])
# X_test_vectorized = vect_product.transform(X_test[Field.product_name_tokenized])
# features = vect_review.get_feature_names()
# print(features)

print(X_train_vectorized.shape)
print(X_test_vectorized.shape)

model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, max_iter=5, random_state=42, n_jobs=-1)
print(y_train.shape)
model.fit(X_train_vectorized, y_train[Field.stars])
# save the model to disk
# filename = 'finalized_model.sav'
# pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)


predictions = model.predict(X_test_vectorized)
print(predictions.shape)
# print(f1_score(y_test[Field.stars], predictions, average='binary'))
print(f1_score(y_test[Field.stars], predictions, average='micro'))
print(f1_score(y_test[Field.stars], predictions, average='macro'))
print(f1_score(y_test[Field.stars], predictions, average='weighted'))
print(precision_score(y_test[Field.stars], predictions, average='weighted'))
print(recall_score(y_test[Field.stars], predictions, average='weighted'))
print(classification_report(y_test[Field.stars], predictions))
# print(f1_score(y_test[Field.stars], predictions, average='samples'))
