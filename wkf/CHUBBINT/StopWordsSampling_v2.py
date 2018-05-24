import math
import random
import re

import numpy as np
import pandas as pd
import sys

from sklearn.feature_extraction.text import CountVectorizer

from wkf.CHUBBINT.Const import File, Column, Category
from wkf.CHUBBINT.vectorizers import NewCountVectorizer


def vectorizer(text):
    return NewCountVectorizer(
        ngram_range=(1, 1),
        stop_words=None,
        preprocessor=None
        # binary=False,
    ).fit(text)


def get_term_counts(text):
    final_df = None
    try:
        vect = vectorizer(text)
        words_df = pd.DataFrame({"term": list(vect.vocabulary_.keys())}, index=vect.vocabulary_.values())
        # transformed = vect.transform([" ".join(text.tolist())])
        transformed = vect.transform(text)
        # df = pd.DataFrame(transformed.todense())
        # df["tmp_col"] = 0
        # df = df.sum()
        coo = transformed.tocoo(copy=False)
        coo = coo.sum(axis=0)
        # transformed = pd.DataFrame({'count': coo.data})
        # print(type(coo))
        # print(np.squeeze(np.asarray(coo)))
        # sys.exit(0)

        words_df["count"] = pd.Series(np.squeeze(np.asarray(coo)))
        # final_df = transformed.join(words_df)
        final_df = words_df
        final_df.sort_values(["count"], inplace=True, ascending=False)
        # final_df["rank"] = final_df["count"].rank(ascending=False, method="dense")
        final_df.set_index("term", inplace=True)
        # final_df.to_csv("tmp.csv", encoding='utf-8', index=False, quotechar='"', sep=',')
    except:
        print(text)
    return final_df


train = pd.read_table(File.train, encoding='utf-8', quotechar='"', sep=',', dtype='str')
# train = pd.read_table(File.data, encoding='utf-8', quotechar='"', sep=',', dtype='str')
test = pd.read_table(File.test, encoding='utf-8', quotechar='"', sep=',', dtype='str')
# train = test

category = Category().HAND
handDocs = train[train[Column.category] == category]
handDocs.to_csv("docs.csv", encoding='utf-8', index=False, quotechar='"', sep=',')
# data = handDocs[Column.desc_tagged]
data = train[Column.desc_tagged]
data.to_csv("data.csv", encoding='utf-8', index=False)
lexicon = get_term_counts(text=data)
lexicon_wt = lexicon
lexicon_wt["mean/max"] = 0
# lexicon["wt"] = 0
# lexicon["wt_rank"] = 0
for category in Category().__dict__.values():
    # for category in [Category().LUNGS]:
    lexicon = lexicon.join(
        get_term_counts(text=train[Column.desc_tagged][train[Column.category] == category]),
        rsuffix=category
    ).fillna(0)
    p_collection = lexicon["count"] / lexicon["count"].sum()
    p_category = lexicon["count" + category] / lexicon["count" + category].sum()
    p_class = lexicon["count" + category] / lexicon["count"]
    p = p_category / p_collection
    # wt = p_category * p.apply(np.log)
    wt =   p_collection / p_collection.max() * p_category / p_category.max() / p_class / p_class.max()
    lexicon["wt" + category] = wt
    lexicon_wt["wt" + category] = wt

lexicon.fillna(0, inplace=True)
lexicon.to_csv("lexicon.csv", encoding='utf-8', index=True, quotechar='"', sep=',')
lexicon_wt.fillna(0, inplace=True)
lexicon_wt["mean/max"] = lexicon_wt.iloc[:, 2:].mean(axis=1) / lexicon_wt.iloc[:, 2:].max(axis=1)
lexicon_wt.sort_values(["mean/max"], inplace=True)
lexicon_wt.to_csv("lexicon_wt.csv", encoding='utf-8', index=True, quotechar='"', sep=',')

# lexicon["p_category"] = lexicon["count" + category] / lexicon["count" + category].sum()
# lexicon["p_collection"] = lexicon["count"] / lexicon["count"].sum()
# lexicon["p"] = lexicon["p_category"] / lexicon["p_collection"]
# lexicon["wt"] = lexicon["p_category"] * lexicon["p"].apply(np.log2)
# lexicon["wt"] = lexicon["wt"] / lexicon["wt"].max() + 1
# lexicon["wt_log"] = lexicon["wt"].apply(np.log2)
sys.exit()

lexicon_terms = []

for term in lexicon.index:
    for i in range(lexicon.loc[term, "count"]):
        lexicon_terms.append(term)

y = 100
for j in range(1, y + 1):
    random.shuffle(lexicon_terms)
    rnd_term = random.choice(lexicon_terms)
    print("rnd_term", j, ":", rnd_term)
    sampled_docs = data[data.apply(lambda x: rnd_term in vectorizer([x]).get_feature_names())]
    sampled_docs_lexicon = get_term_counts(text=sampled_docs)
    # sampled_docs_lexicon.to_csv("tmp.csv", encoding='utf-8', index=False, quotechar='"', sep=',')

    col_name = str(j) + "_wt"
    lexicon[col_name] = 0
    for term in sampled_docs_lexicon.index:
        term_freq = sampled_docs_lexicon.loc[term, "count"]

        docs_sample_len = sampled_docs_lexicon["count"].sum()
        tokens_count = lexicon["count"].sum()
        p_x = term_freq / docs_sample_len
        p_c = term_freq / tokens_count
        w_t = p_x * math.log2(p_x / p_c)
        lexicon.loc[term, col_name] = w_t
    lexicon[col_name] = lexicon[col_name] / lexicon[col_name].max()
    lexicon["wt"] = lexicon["wt"] + lexicon[col_name] / y
    lexicon[col_name + "_rank"] = lexicon[col_name].rank(ascending=True, method="dense")

lexicon.to_csv("lexicon.csv", encoding='utf-8', index=True, quotechar='"', sep=',')
