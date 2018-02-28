import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from profitero_data_scientist.Model import Model
from profitero_data_scientist.Vectorizer import Vectorizer
from profitero_data_scientist.utils.Constants import File, Field, Language, Models, Vectorizers
from profitero_data_scientist.utils.Data import Data


# prepare vectorizers



# df = Data(File.jd_reviews, Language.ch).df
df = pd.read_table(File.jd_reviews, encoding='utf-8', quotechar='"', sep=',', dtype='str')
df.dropna(inplace=True)
# df = df.sample(frac=0.01, random_state=10)

# X_train = df[[Field.review, Field.created_at, Field.product_name, Field.product_category]]
X_train = df[[Field.review, Field.created_at, Field.product_name]]
Y_test = df[[Field.stars]]

# X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_test, stratify=Y_test, test_size=0.15, random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_test)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
# print(X_train.head())

# vect = CountVectorizer().fit(X_train[Field.review_en])
# vect = CountVectorizer(min_df=3, ngram_range=(3, 4)).fit(X_train[predictor])
# vect_review = CountVectorizer(ngram_range=(2, 3)).fit(X_train[Field.review_tokenized])
# vect_product = CountVectorizer(ngram_range=(2, 3)).fit(X_train[Field.product_name_tokenized])
# vect_review = CountVectorizer(
#     # input='content',
#     # encoding='utf-8',
#     # decode_error='strict',
#     # strip_accents=None,
#     # lowercase=True,
#     # preprocessor=None,
#     # tokenizer=None,
#     # stop_words='english',
#     # token_pattern=r"(?u)\b\w\w+\b",
#     ngram_range=(1, 5),
#     # analyzer='word',
#     # max_df=1.0,
#     # min_df=1,
#     # max_features=None,
#     # vocabulary=None,
#     # binary=True,
#     # dtype=np.int64
# ).fit(X_train[Field.review])

# vect_product = CountVectorizer(
#     input='content',
#     encoding='utf-8',
#     decode_error='strict',
#     strip_accents=None,
#     lowercase=True,
#     preprocessor=None,
#     tokenizer=None,
#     stop_words='english',
#     token_pattern=r"(?u)\b\w\w+\b",
#     ngram_range=(1, 2),
#     analyzer='word',
#     max_df=1.0,
#     min_df=1,
#     max_features=5,
#     vocabulary=None,
#     binary=True,
#     dtype=np.int64
# ).fit(X_train[Field.product_name])

# vect = TfidfVectorizer(min_df=5).fit(X_train[predictor])


X_train_vectorized = Vectorizer(Vectorizers.ch_revi_1_5_ngram_count).transform(X_train[Field.review])
tmp = Vectorizer(Vectorizers.ch_prod_1_2_ngram_count).transform(X_train[Field.product_name])
X_train_vectorized = hstack((X_train_vectorized, tmp))
# X_train_vectorized = hstack((X_train_vectorized, np.array(X_train[Field.created_at])[:, None]))
# X_train_vectorized = hstack((X_train_vectorized, np.array(X_train[Field.product_category])[:, None]))

X_test_vectorized = Vectorizer(Vectorizers.ch_revi_1_5_ngram_count).transform(X_test[Field.review])
tmp = Vectorizer(Vectorizers.ch_prod_1_2_ngram_count).transform(X_test[Field.product_name])
X_test_vectorized = hstack((X_test_vectorized, tmp))
# X_test_vectorized = hstack((X_test_vectorized, np.array(X_test[Field.created_at])[:, None]))
# X_test_vectorized = hstack((X_test_vectorized, np.array(X_test[Field.product_category])[:, None]))

print(X_train_vectorized.shape)
print(X_test_vectorized.shape)

# ***********************************************************************
model = Model(Models.model_1_sgd)
# model = Model(Models.model_2_mpl)
model.train(X_train_vectorized, Y_train.values.ravel())
model.extract(X_test_vectorized, Y_test)
# ***********************************************************************
