import pandas as pd
import matplotlib.pyplot as plt
from wkf.CHUBBINT.Const import File, Column
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import re
from collections import OrderedDict
from operator import itemgetter

df = pd.read_table(File.data_body, encoding='utf-8', quotechar='"', sep=',', dtype='str')
df = df.dropna().drop_duplicates()

stop_words=[line.strip() for line in open(File.stop_words, "r", encoding="utf-8")]
stop_words.extend([""])

def split(text):
    words_list = re.split("[^0-9a-wA-W]", text.lower())
    words_list_new = []
    for word in words_list:
        if word not in stop_words:
            words_list_new.append(word)
    return words_list_new


words = set()


def add_words(text):
    global words
    for word in split(text):
        # if word not in ["", "a", "the", "and", "ee", "in", "to", "of", "on"]:
        words.add(word)

    # df[Col.desc_tagged].apply(lambda text: next(words.add(word.lower()) for word in text.split()))


# df[Col.desc_tagged].apply(lambda text: next(words.add(word.lower()) for word in re.split("(?u)\b\w\w+\b", text)))
df[Column.desc_tagged].apply(lambda text: add_words(text))

for word in words:
    print(word)

print(len(words))


def fill_dict(text):
    # {"a":"s"}.
    global dictionary
    for word in split(text):
        if word in dictionary.keys():
            dictionary[word] += 1
        else:
            dictionary[word] = 1
    return dictionary


for category in df[Column.category].drop_duplicates().dropna():
    dictionary = {}
    df_1 = df[df[Column.category] == category]
    df_1["words"] = df_1[Column.desc_tagged].apply(lambda text: fill_dict(text))

    dictionary = OrderedDict(sorted(dictionary.items(), key=itemgetter(1), reverse=True))
    new_dict = OrderedDict()
    for word in dictionary.keys():
        if dictionary.get(word) > 2:
            new_dict[word] = dictionary.get(word)
    print(new_dict)
    print("cat:", category, "size:", df_1.shape[0], "dict:", len(new_dict))
