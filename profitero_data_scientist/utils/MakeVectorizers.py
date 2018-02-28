import pandas as pd

from profitero_data_scientist.Vectorizer import Vectorizer
from profitero_data_scientist.utils.Constants import File, Vectorizers, Field

df_ch = pd.read_table(File.jd_reviews, encoding='utf-8', quotechar='"', sep=',', dtype='str')
df_ch.dropna(inplace=True)

Vectorizer(Vectorizers.ch_revi_1_5_ngram_count).fit(df_ch[Field.review])
Vectorizer(Vectorizers.ch_prod_1_2_ngram_count).fit(df_ch[Field.product_name])




