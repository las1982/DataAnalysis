import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
# from nltk.parse.corenlp import CoreNLPTokenizer
import os


# df = pd.read_table('jd_reviews.csv', encoding='utf-8', quotechar='"', sep=',', dtype='str', na_values=["nan"], keep_default_na=False)
df = pd.read_table('jd_reviews.csv', encoding='utf-8', quotechar='"', sep=',', dtype='str')

# print(df.head())
# print(df.shape)
# print(df.describe())

df.dropna(inplace=True)
# print(df.head())
# print(df.shape)
# print(df.describe())

# X = df[['review', 'created_at', 'product_id', 'product_name']]
X = df[['review']]
y = df[['stars']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# cmap = cm.get_cmap('gnuplot')
# scatter = pd.plotting.scatter_matrix(X_train, c=y_train, marker='o', s=40, hist_kwds={'bins': 15}, figsize=(9, 9),
#                                      cmap=cmap)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X_train['width'], X_train['height'], X_train['color_score'], c=y_train, marker='o', s=100)
# ax.set_xlabel('width')
# ax.set_ylabel('height')
# ax.set_zlabel('color_score')
# plt.show()

# segmenter = StanfordSegmenter(
#     path_to_jar="/home/alex/Downloads/stanford-segmenter-2017-06-09/stanford-segmenter-3.8.0.jar",
#     path_to_slf4j="/home/alex/Downloads/stanford-segmenter-2017-06-09/slf4j-1.7.25/slf4j-api-1.7.25.jar",
#     path_to_sihan_corpora_dict="/home/alex/Downloads/stanford-segmenter-2017-06-09/data",
#     path_to_model="/home/alex/Downloads/stanford-segmenter-2017-06-09/data/pku.gz",
#     path_to_dict="/home/alex/Downloads/stanford-segmenter-2017-06-09/data/dict-chris6.ser.gz"
# )

segmenter = StanfordSegmenter(
    path_to_jar="/home/alex/Downloads/stanford-segmenter-2017-06-09/stanford-segmenter-3.8.0.jar",
    path_to_slf4j="/home/alex/Downloads/stanford-segmenter-2017-06-09/slf4j-1.7.25/slf4j-api-1.7.25.jar",
    path_to_sihan_corpora_dict="/home/alex/Downloads/stanford-segmenter-2017-06-09/data",
    path_to_model="/home/alex/Downloads/stanford-segmenter-2017-06-09/data/pku.gz",
    path_to_dict="/home/alex/Downloads/stanford-segmenter-2017-06-09/data/dict-chris6.ser.gz"
)
segmenter.default_config('zh')

# corpus = ['这是一个最好的时代', '中国好声音']
# vectorizer = CountVectorizer(tokenizer=lambda text: segmenter.segment(text).split())
# X = vectorizer.fit_transform(corpus)
# features = vectorizer.get_feature_names()
# print(features)

# sent = u'这是斯坦福中文分词器测试'
# print(segmenter.segment(sent))


# vect = CountVectorizer(tokenizer=lambda text: segmenter.segment(text).split()).fit(X_test.review)
vect = CountVectorizer(tokenizer=lambda text: segmenter.segment(text).split())
# vect = CountVectorizer()

X_train_vectorized = vect.fit_transform(X_test.review)
# X_train_vectorized = vect.transform(X_test)
features = vect.get_feature_names()
print(features)

# X_test_vectorized = vect.transform(X_test)
# print(X_train_vectorized, X_test_vectorized)
#
# model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, max_iter=5, random_state=42, n_jobs=-1)
# model.fit(X_train_vectorized, y_train)
# predictions = model.predict(X_test_vectorized)
