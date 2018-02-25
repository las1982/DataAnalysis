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

filename = positiveFiles[110]
with codecs.open(filename, "r") as doc_file:
    text = doc_file.read()
    text = text.replace("\n", "")
    text = text.replace("\r", "")
print("==Orginal==:\n\r{}".format(text))

stopwords = [line.rstrip() for line in codecs.open('./data/chinese_stop_words.txt', "r", encoding="utf-8")]
seg_list = jieba.cut(text, cut_all=False)
final = []
seg_list = list(seg_list)
for seg in seg_list:
    if seg not in stopwords:
        final.append(seg)
print("==Tokenized==\tToken count:{}\n\r{}".format(len(seg_list), " ".join(seg_list)))
print("==Stop Words Removed==\tToken count:{}\n\r{}".format(len(final), " ".join(final)))