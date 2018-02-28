import pickle
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Dense
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
import pandas as pd
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
from profitero_data_scientist.utils.Constants import Field
from profitero_data_scientist.utils.Constants import File
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


X_train_vectorized = vect_review.transform(X_train[Field.review_tokenized])
# X_train_vectorized = pd.DataFrame(vect_review.transform(X_train[Field.review_tokenized]).toarray())
# X_train_vectorized = X_train_vectorized.merge(
#     pd.DataFrame(vect_product.transform(X_train[Field.product_name_tokenized]).toarray()), left_index=True,
#     right_index=True, how='left')
# X_train_vectorized[Field.created_at_day_of_week] = X_train[Field.created_at_day_of_week].values

X_test_vectorized = vect_review.transform(X_test[Field.review_tokenized])
# X_test_vectorized = pd.DataFrame(vect_review.transform(X_test[Field.review_tokenized]).toarray())
# X_test_vectorized = X_test_vectorized.merge(
#     pd.DataFrame(vect_product.transform(X_test[Field.product_name_tokenized]).toarray()), left_index=True,
#     right_index=True, how='left')
# X_test_vectorized[Field.created_at_day_of_week] = X_test[Field.created_at_day_of_week].values

# X_test_vectorized = vect_review.transform(X_test[Field.review_tokenized])
# X_test_vectorized = vect_product.transform(X_test[Field.product_name_tokenized])
features = vect_review.get_feature_names()
# print(features)

print(X_train_vectorized.shape)
print(X_test_vectorized.shape)

embedding_dim = 256
maxLength = X_train_vectorized.shape[1]
output_dimen = y_train.shape[1]

model = Sequential()
model.add(Embedding(len(features), embedding_dim,input_length = maxLength))
# Each input would have a size of (maxLength x 256) and each of these 256 sized vectors are fed into the GRU layer one at a time.
# All the intermediate outputs are collected and then passed on to the second GRU layer.
model.add(GRU(256, dropout=0.9, return_sequences=True))
# Using the intermediate outputs, we pass them to another GRU layer and collect the final output only this time
model.add(GRU(256, dropout=0.9))
# The output is then sent to a fully connected layer that would give us our final output_dim classes
model.add(Dense(output_dimen, activation='softmax'))
# We use the adam optimizer instead of standard SGD since it converges much faster
tbCallBack = TensorBoard(log_dir='./Graph/sentiment_chinese', histogram_freq=0,
                            write_graph=True, write_images=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(X_train_vectorized, y_train, validation_split=0.1, batch_size=32, epochs=1, verbose=1, callbacks=[tbCallBack])
model.save('./profitero_data_scientist/data/sentiment_chinese_model.hdf5')

print("Saved model!")

sys.exit(0)
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
