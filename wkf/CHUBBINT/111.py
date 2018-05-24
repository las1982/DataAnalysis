import os

import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab as pl
import sys
import re
from wkf.CHUBBINT.Const import File, Column

# df = pd.read_table("/home/alex/work/projects/DataAnalysis/wkf/CHUBBINT/features.csv", encoding='utf-8',
#                    quotechar='"', sep='\t', dtype='str')
# df.drop_duplicates(inplace=True)
# df["feature"] = df["feature"].str.replace("[0-9A-Z]+", "", n=-1, case=True, flags=0).str.rstrip(to_strip="_")
# df = df.groupby("file")["feature"].apply(lambda feature: " ".join(feature))
# df.to_csv("/home/alex/work/projects/DataAnalysis/wkf/CHUBBINT/features_to_use.csv", encoding='utf-8', index=False)

train = pd.read_table(File.train, encoding='utf-8', quotechar='"', sep=',', dtype='str')
test = pd.read_table(File.test, encoding='utf-8', quotechar='"', sep=',', dtype='str')
vect = TfidfVectorizer(
    input='content',
    encoding='utf-8',
    decode_error='strict',
    strip_accents=None,
    lowercase=True,
    preprocessor=None,
    tokenizer=None,
    analyzer='word',
    stop_words=None,
    token_pattern="[a-z]{2,}",
    ngram_range=(1, 3),
    max_df=1.0, min_df=1,
    max_features=None,
    vocabulary=None,
    binary=False,
    dtype=np.int64,
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False
)
vect.fit(train[Column.desc_tagged])
transformed_train = vect.transform(train[Column.desc_tagged])
tfidf_train = pd.DataFrame(transformed_train.A, columns=vect.get_feature_names())
# tfidf_train.to_csv("tfidf_train.csv")

transformed_test = vect.transform(test[Column.desc_tagged])
tfidf_test = pd.DataFrame(transformed_test.A, columns=vect.get_feature_names())
# tfidf_test.to_csv("tfidf_test.csv")

sys.exit(0)
coo = transformed.tocoo(copy=False)
# coo = coo.sum(axis=0)

words_df = pd.DataFrame(np.squeeze(np.asarray(coo)))


# def normalized(str):
#     return re.sub("[^0-9a-z]", "_", str.lower())
#
#
# for index in range(df.shape[0]):
#     label = re.sub("[^0-9a-z]", "_", df.loc[index, Column.category].lower())
#     file_name = "/home/alex/work/projects/DataAnalysis/wkf/CHUBBINT/labeled/" + label + "/" + str(index) + ".txt"
#     os.makedirs(os.path.dirname(file_name), exist_ok=True)
#     with open(file_name, "x") as file:
#         file.write(df.loc[index, Column.desc_tagged])
