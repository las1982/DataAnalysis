import pandas as pd
import matplotlib.pyplot as plt
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

stop_words = list(ENGLISH_STOP_WORDS)
# for line in open(File.stop_words, "r", encoding="utf-8"):
#     stop_words.append(line.strip())

# category = Category.SKULL
# print(Category.__dict__)
labels = [0, 1]
ngram_range = (1, 3)
print("label\tcategory\ttn\tfp\tfn\ttp\tsupport\tP\tR\tF1")
tp_cnt = 0
fp_cnt = 0
tn_cnt = 0
fn_cnt = 0
for category in Category().__dict__.keys():
    category = Category().__dict__.get(category)
    # vect = CountVectorizer(
    # vect = TfidfVectorizer(
    vect = NewCountVectorizer(
        # input='content',
        # encoding='utf-8',
        # decode_error='strict',
        # strip_accents=None,
        lowercase=True,
        preprocessor=lambda text: re.sub("\d+", "", text.lower()),
        # tokenizer=None,
        # stop_words=stop_words,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=ngram_range,
        analyzer='word',
        # max_df=0.75,
        # min_df=1,
        # max_features=10000,
        # vocabulary=None,
        binary=True,
        dtype=np.int64
    )
    # print(category)
    # print(train[Column.desc_tagged][train[category] == '1'])
    # sys.exit()
    bag = vect.fit(train[Column.desc_tagged][train[category] == "1"])
    # bag = vect.fit(train[Column.desc_tagged])
    # print("stop words:", vect.get_stop_words().__len__(), vect.get_stop_words())
    # print("features:", vect.get_feature_names().__len__(), vect.get_feature_names())
    # print(bag)

    category_vocab = vect.vocabulary_
    # print("vocabulary:", category_vocab.__len__())

    other_vocab = vect.fit(train[Column.desc_tagged][train[category] == "0"]).vocabulary_
    # print("other vocabulary:", other_vocab.__len__())

    new_category_vocab = category_vocab
    # new_category_vocab = {}
    # i = 0
    # for key in category_vocab.keys():
    #     if key not in other_vocab.keys():
    #         new_category_vocab[key] = i
    #         i += 1
    # print("new vocabulary:", new_category_vocab.__len__())

    vect = NewCountVectorizer(
        # vect=TfidfVectorizer(
        # input='content',
        # encoding='utf-8',
        # decode_error='strict',
        # strip_accents=None,
        lowercase=True,
        # preprocessor=lambda text: re.sub("\d+", "", text.lower()),
        # tokenizer=None,
        # stop_words=stop_words,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=ngram_range,
        analyzer='word',
        # max_df=0.75,
        # min_df=1,
        # max_features=10000,
        vocabulary=new_category_vocab,
        binary=True,
        dtype=np.int64
    )
    vect = vect.fit(train[Column.desc_tagged][train[category] == "1"])
    # print("new vocabulary:", vect.vocabulary_.__len__())

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
