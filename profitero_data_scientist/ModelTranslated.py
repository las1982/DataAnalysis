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

# df = pd.read_table('jd_reviews.csv', encoding='utf-8', quotechar='"', sep=',', dtype='str', na_values=["nan"], keep_default_na=False)
from profitero_data_scientist.utils.Enums import Field
from profitero_data_scientist.utils.Enums import File

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
df = df.sample(frac=0.5, random_state=10)

# print(df[Field.stars].describe())
# print(df.groupby(Field.stars).agg({Field.review: 'count'}))

X = df[[Field.review_en]]
y = df[[Field.stars]]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# vect = CountVectorizer().fit(X_train[Field.review_en])
# vect = CountVectorizer(min_df=3, ngram_range=(3, 4)).fit(X_train[Field.review_en])
vect = CountVectorizer(ngram_range=(1, 5)).fit(X_train[Field.review_en])
# vect = TfidfVectorizer(min_df=5).fit(X_train[Field.review_en])


X_train_vectorized = vect.transform(X_train[Field.review_en])
X_test_vectorized = vect.transform(X_test[Field.review_en])
features = vect.get_feature_names()
print(features)

print(X_train_vectorized.shape)
print(X_test_vectorized.shape)

# model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, max_iter=5, random_state=42, n_jobs=-1)
# model = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='kd_tree', n_jobs=-1)
model = NearestNeighbors(n_neighbors=1, n_jobs=-1)
print(y_train.shape)
model.fit(X_train_vectorized, y_train[Field.stars])
print(model.kneighbors(X_test_vectorized, return_distance=False))
print(len(model.kneighbors(X_test_vectorized, return_distance=False)))

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
