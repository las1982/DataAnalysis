import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from wkf.CHUBBINT.Const import File, Column

df = pd.read_table(File.data_body, encoding='utf-8', quotechar='"', sep=',', dtype='str')
df_train, df_test = train_test_split(
    df,
    shuffle=True,
    stratify=df[Column.category],
    # random_state=42,
    test_size=0.25)
df_train = df_train.drop_duplicates().dropna()
df_test = df_test.drop_duplicates().dropna()

vect = CountVectorizer(ngram_range=(1, 3))
data = vect.fit_transform(df[Column.desc_tagged])


model = KMeans(n_clusters=10, n_jobs=-1, verbose=True)
model.fit(data)
predictions = model.predict(data)
df["cluster"] = predictions
df.to_csv("clusters.csv", encoding='utf-8', index=False, quotechar='"', sep=',')
