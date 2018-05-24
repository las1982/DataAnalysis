import math
import random
import re

import numpy as np
import pandas as pd

from wkf.CHUBBINT.Const import File, Column, Category
from wkf.CHUBBINT.vectorizers import NewCountVectorizer


def vectorizer(text):
    return NewCountVectorizer(
        ngram_range=(1, 1),
        binary=False,
    ).fit(text)


def get_term_counts(text):
    final_df = None
    try:
        vect = vectorizer(text)
        words_df = pd.DataFrame({"term": list(vect.vocabulary_.keys())}, index=vect.vocabulary_.values())
        transformed = vect.transform([" ".join(text.tolist())])
        coo = transformed.tocoo(copy=False)
        transformed = pd.DataFrame({'count': coo.data})
        final_df = transformed.join(words_df)
        final_df.sort_values(["count"], inplace=True, ascending=False)
        final_df["rank"] = final_df["count"].rank(ascending=False, method="dense")
        final_df.set_index("term", inplace=True)
        # final_df.to_csv("tmp.csv", encoding='utf-8', index=False, quotechar='"', sep=',')
    except:
        print(text)
    return final_df


train = pd.read_table(File.train, encoding='utf-8', quotechar='"', sep=',', dtype='str')
test = pd.read_table(File.test, encoding='utf-8', quotechar='"', sep=',', dtype='str')

handDocs = train[train[Column.category] == Category().HAND]
handDocs.to_csv("docs.csv", encoding='utf-8', index=False, quotechar='"', sep=',')
# data = handDocs[Column.desc_tagged]
data = train[Column.desc_tagged]
data.to_csv("data.csv", encoding='utf-8', index=False)

lexicon = get_term_counts(text=data)
lexicon["wt"] = 0
lexicon["wt_rank"] = 0

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
