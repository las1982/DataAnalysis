import pandas as pd
import matplotlib.pyplot as plt
from wkf.CHUBBINT.Const import File, Column, Models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import numpy as np
# from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import re
import sys
from sklearn.feature_extraction import image

# print(ENGLISH_STOP_WORDS.__len__())
# sys.exit(0)

df = pd.read_table(File.data_body, encoding='utf-8', quotechar='"', sep=',', dtype='str')
df = df.dropna().drop_duplicates()
x_train, x_test, y_train, y_test = train_test_split(
    df[Column.desc_tagged],
    df[Column.category],
    shuffle=True,
    stratify=df[Column.category],
    random_state=42,
    test_size=0.25)
print("x:", x_train.shape)
print("y:", y_train.shape)

stop_words = list(ENGLISH_STOP_WORDS)
for line in open(File.stop_words, "r", encoding="utf-8"):
    stop_words.append(line.strip())

vect = CountVectorizer(
    input='content',
    encoding='utf-8',
    decode_error='strict',
    strip_accents=None,
    lowercase=True,
    preprocessor=lambda text: re.sub("[^a-z]+", " ", text.lower()),
    tokenizer=None,
    stop_words=stop_words,
    token_pattern=r"(?u)\b\w\w+\b",
    ngram_range=(1, 3),
    analyzer='word',
    max_df=0.75,
    min_df=1,
    # max_features=10000,
    vocabulary=None,
    binary=True,
    dtype=np.int64
)
vect.fit(x_train)
print(vect.get_feature_names())
print(len(vect.get_feature_names()))

x_train = vect.transform(x_train)
x_test = vect.transform(x_test)
print("x vect:", x_train.shape)
print("y vect:", y_train.shape)

# model = Models.SGDClassifier
model = Models.LogisticRegression
# model = Models.LogisticRegressionCV

model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(classification_report(y_test, predictions))

df = pd.DataFrame()
df["gold"] = y_test
df["pred"] = predictions

print(df)

print(pd.Series(y_test == predictions).sum())