from datetime import datetime
from profitero_data_scientist.utils.Data import categorize_product

import pandas as pd

from profitero_data_scientist.utils.Constants import File, Field

df = pd.read_table(File.jd_reviews, encoding='utf-8', quotechar='"', sep=',', dtype='str')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df1 = df[[Field.stars, Field.created_at, Field.review_id]]
df1 = df.groupby([Field.stars]).count()
print(df1)
df1.to_csv('stars.csv', encoding='utf-8', index=True, quotechar='"', sep=',')

df2 = df[[Field.stars, Field.created_at, Field.review_id]]
df2[Field.created_at] = df2[Field.created_at].apply(lambda date: datetime.strptime(date, '%Y-%m-%d').weekday())
df2 = df2.groupby([Field.stars, Field.created_at]).count()
print(df2)
df2.to_csv('weekday.csv', encoding='utf-8', index=True, quotechar='"', sep=',')

df3 = df[[Field.stars, Field.created_at, Field.review_id]]
df3[Field.created_at] = df3[Field.created_at].apply(lambda date: datetime.strptime(date, '%Y-%m-%d').month)
df3 = df3.groupby([Field.stars, Field.created_at]).count()
print(df3)
df3.to_csv('month.csv', encoding='utf-8', index=True, quotechar='"', sep=',')

df4 = df[[Field.product_name, Field.review_id]]
df4 = categorize_product(df4, Field.product_name)
df4 = df4.groupby([Field.product_name]).count()
print(df4)
df4.to_csv('product.csv', encoding='utf-8', index=True, quotechar='"', sep=',')
