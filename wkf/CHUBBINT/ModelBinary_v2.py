import pandas as pd
import matplotlib.pyplot as plt

from wkf.CHUBBINT import Stop
from wkf.CHUBBINT.Const import File, Category, Column, Models
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import re
import sys
from sklearn.feature_extraction import image
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support

# print(ENGLISH_STOP_WORDS.__len__())
from wkf.CHUBBINT.vectorizers import NewCountVectorizer

train = pd.read_table(File.train, encoding='utf-8', quotechar='"', sep=',', dtype='str')
test = pd.read_table(File.test, encoding='utf-8', quotechar='"', sep=',', dtype='str')

# train = pd.concat([train, pd.get_dummies(
#     train,
#     prefix="",
#     prefix_sep="",
#     dummy_na=False,
#     columns=[Column.category],
#     sparse=False,
#     drop_first=False
# ).iloc[:, 1:]], axis=1)
# train.to_csv(File.train + ".binary", encoding='utf-8', index=False, quotechar='"', sep=',')
#
# test = pd.concat([test, pd.get_dummies(
#     test,
#     prefix="",
#     prefix_sep="",
#     dummy_na=False,
#     columns=[Column.category],
#     sparse=False,
#     drop_first=False
# ).iloc[:, 1:]], axis=1)
# test.to_csv(File.test + ".binary", encoding='utf-8', index=False, quotechar='"', sep=',')
# sys.exit(0)

labels = [0, 1]
ngram_range = (1, 4)
print("label\tcategory\ttn\tfp\tfn\ttp\tsupport\tP\tR\tF1")
tp_cnt = 0
fp_cnt = 0
tn_cnt = 0
fn_cnt = 0
# for category in Category().__dict__.keys():
stop_ngrams = [
    "cut",
    "finger",
    "report",
    "knife",
    "slipped",
    "went",
    "door",
    "cutting",
    "box",
    "fell",
    "thumb",
    "caused",
    "hit",
    "laceration",
    "car",
    "palm",
    "caught",
    "working",
    "pain",
    "holding",
    "stated",
    "got",
    "started",
    "get",
    "cut hand",
    "work",
    "drill",
    "metal",
    "glass",
    "top",
    "getting",
    "cat",
    "area",
    "piece",
    "causing",
    "machine",
    "put",
    "sustained",
    "back",
    "cut right",
    "bit",
    "hit hand",
    "burned",
    "part",
    "injury",
    "hurt",
    "cut left",
    "using",
    "came",
    "glove",
    "coffee",
    "opening",
    "moving",
    "cart",
    "cut left hand",
    "stuck",
    "dog"
]


def analize(doc):
    analizer = NewCountVectorizer(ngram_range=ngram_range).build_analyzer()
    # return lambda doc: rrr(tokenizer, doc)
    words = [ngram for ngram in analizer(doc)]
    # words = (ngram for ngram in analizer(doc) if ngram not in Stop.stop_ngrams.get(Category().HAND))
    words = (ngram for ngram in analizer(doc) if ngram not in stop_ngrams)
    # return lambda doc: (lemmatizer.lemmatize(w) for w in self.tokenizer(doc) if w not in self.stop)
    return words


for category in [Category().HAND]:
    vect = NewCountVectorizer(
        ngram_range=ngram_range,
        # analyzer='word',
        analyzer=lambda doc: analize(doc),
        binary=True,
    )
    vect = vect.fit(train[Column.desc_tagged][train[category] == "1"])
    print("new vocabulary:", vect.vocabulary_.__len__())
    print("features:", vect.get_feature_names().__len__(), vect.get_feature_names())

    x_train = vect.transform(train[Column.desc_tagged])
    x_test = vect.transform(test[Column.desc_tagged])
    y_train = train[category]
    y_test = test[category]
    # y_train = train[Column.category]
    # y_test = test[Column.category]

    # print("x vect:", x_train.shape)
    # print("y vect:", y_train.shape)

    model = Models.SGDClassifier
    # model = Models.DecisionTreeClassifier
    # model = Models.LogisticRegression

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    # print(classification_report(y_test, predictions))
    f1 = f1_score(
        y_true=y_test,
        y_pred=predictions,
        labels=labels,
        pos_label=1,
        average=None,
        sample_weight=None
    )
    p = precision_score(
        y_true=y_test,
        y_pred=predictions,
        labels=labels,
        pos_label=1,
        average=None,
        sample_weight=None
    )
    r = recall_score(
        y_true=y_test,
        y_pred=predictions,
        labels=labels,
        pos_label=1,
        average=None,
        sample_weight=None
    )
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    p, r, f1, support = precision_recall_fscore_support(
        y_true=y_test,
        y_pred=predictions,
        labels=labels
    )
    for label in labels:
        print(
            "\t".join(str(x) for x in [label, category, tn, fp, fn, tp, support[label], p[label], r[label], f1[label]]))
    tp_cnt += tp
    fp_cnt += fp
    tn_cnt += tn
    fn_cnt += fn
    # print(confusion_matrix(y_test, predictions))
print("tp\tfp\ttn\tfn")
print("\t".join([str(tp_cnt), str(fp_cnt), str(tn_cnt), str(fn_cnt)]))
