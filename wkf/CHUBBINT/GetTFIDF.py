import sys
import time

import gensim

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

type_of_data = sys.argv[1]
data_file = "/home/alex/work/wf/CHUBBINT/detailed_part_of_body/data/train.csv"
train = pd.read_table(data_file, encoding='utf-8', quotechar='"', sep=',', dtype='str')
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
    token_pattern="[a-z]+",
    ngram_range=(1, 3),
    max_df=1.0,
    min_df=1,
    max_features=None,
    vocabulary=None,
    binary=False,
    dtype=np.int64,
    norm='l2',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=False
)
vect.fit(train["desc_tagged"])

data_file = "/home/alex/work/wf/CHUBBINT/detailed_part_of_body/data/" + type_of_data + ".csv"
df = pd.read_table(data_file, encoding='utf-8', quotechar='"', sep=',', dtype='str')
transformed = vect.transform(df["desc_tagged"])
tfidf = pd.DataFrame(transformed.A, columns=vect.get_feature_names())
model = gensim.models.KeyedVectors.load_word2vec_format('/home/alex/work/projects/DataAnalysis/wkf/CHUBBINT/text.model.bin', binary=True)

i = 0
args = [0, 0]
while len(args) == 2:
    if i == 1000:
        time.sleep(3)
    args = eval(input())
    file = args[0]
    word = args[1]

    # file = 5636
    # word = "pain and"

    vector = model.wv.get_vector(word.replace(" ", "_"))

    try:
        score = tfidf.loc[int(file), word]
        word_mul_tfidf_vect = vector * score
        print(",".join([str(val) for val in word_mul_tfidf_vect]))
    except:
        print(0.0)
