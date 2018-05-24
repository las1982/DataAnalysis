import pandas as pd
import matplotlib.pyplot as plt
from wkf.CHUBBINT.Const import File, Column
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import re
import sys
from sklearn.feature_extraction import image

# print(ENGLISH_STOP_WORDS.__len__())
# sys.exit(0)

df = pd.read_table(File.data_body, encoding='utf-8', quotechar='"', sep=',', dtype='str')
df = df.dropna().drop_duplicates()

stop_words = list(ENGLISH_STOP_WORDS)
for line in open(File.stop_words, "r", encoding="utf-8"):
    stop_words.append(line.strip())

# vect = CountVectorizer(
#     input='content',
#     encoding='utf-8',
#     decode_error='strict',
#     strip_accents=None,
#     lowercase=True,
#     preprocessor=lambda text: re.sub("[^a-z]+", " ", text.lower()),
#     tokenizer=None,
#     stop_words=[line.strip() for line in open(Fl.stop_words, "r", encoding="utf-8")].extend(ENGLISH_STOP_WORDS),
#     token_pattern=r"(?u)\b\w\w+\b",
#     ngram_range=(1, 4),
#     analyzer='word',
#     max_df=0.1,
#     min_df=1,
#     # max_features=10000,
#     vocabulary=None,
#     binary=True,
#     dtype=np.int64
# )

vect = CountVectorizer(stop_words=stop_words)
bag = vect.fit(df[Column.desc_tagged])
print("stop words:", vect.get_stop_words())
print("features:", vect.get_feature_names().__len__(), vect.get_feature_names())
print(bag)
bag = vect.transform(df[Column.desc_tagged])
print(bag)
print(vect.vocabulary_)

