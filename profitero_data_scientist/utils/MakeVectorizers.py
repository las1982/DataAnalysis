from profitero_data_scientist.utils.Vectorizer import Vectorizer
from profitero_data_scientist.utils.Data import Data
from profitero_data_scientist.utils.Constants import File, Vectorizers, Field

df_ch = Data(File.jd_reviews).get_original_df()
df_ch.dropna(inplace=True)

Vectorizer(Vectorizers.ch_revi_1_5_ngram_count).fit(df_ch[Field.review])
Vectorizer(Vectorizers.ch_prod_1_3_ngram_count).fit(df_ch[Field.product_name])

df_en = Data(File.jd_reviews).get_translated_df()
df_en.dropna(inplace=True)

Vectorizer(Vectorizers.en_revi_1_5_ngram_count).fit(df_en[Field.review])
Vectorizer(Vectorizers.en_prod_1_2_ngram_count).fit(df_en[Field.product_name])
