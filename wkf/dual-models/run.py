import os
from bs4 import BeautifulSoup
from const import F, M, N
from collections import Counter
import pandas as pd


f = F().f[M.db]
tag_names_counter = Counter()
files = os.listdir(f[N.train_dir])

for file_name in files:
    with open(f[N.train_dir] + file_name, "r") as file:
        soup = BeautifulSoup(file.read(), 'html.parser')
    extraction_tags = soup.find_all(class_="extraction-tag")
    tag_names = set()
    for extraction_tag in extraction_tags:
        tag_names.add(extraction_tag.name)
    tag_names_counter.update(tag_names)
    # break
print(dict(tag_names_counter))

df = pd.DataFrame(index=tag_names_counter.keys())
df["cnt"] = 0
df["percent"] = 0
df["classification_F1"] = 0
df["base_TP"] = 0
df["base_FP"] = 0
df["base_F1"] = 0

df["dual_TP"] = 0
df["dual_FP"] = 0
df["dual_F1"] = 0

df["100_TP"] = 0
df["100_FP"] = 0
df["100_F1"] = 0

df["100_th_TP"] = 0
df["100_th_FP"] = 0
df["100_th_F1"] = 0

df["100_cl_TP"] = 0
df["100_cl_FP"] = 0
df["100_cl_F1"] = 0

for tag_name in tag_names_counter.keys():
    df.loc[tag_name, "cnt"] = tag_names_counter.get(tag_name)

    f1_stats_df = pd.read_csv(f[N.classif_models_dir] + tag_name + "/output/stp/stats/per-tag-stats.csv", index_col=0)
    df.loc[tag_name, "classification_F1"] = f1_stats_df.loc[tag_name, "F1"]

    f1_stats_df = pd.read_csv(f[N.base_stp], index_col=0)
    df.loc[tag_name, "base_TP"] = f1_stats_df.loc[tag_name, "TP"]
    df.loc[tag_name, "base_FP"] = f1_stats_df.loc[tag_name, "FP"]
    df.loc[tag_name, "base_F1"] = f1_stats_df.loc[tag_name, "F1"]

    f1_stats_df = pd.read_csv(f[N.dual_stp], index_col=0)
    df.loc[tag_name, "dual_TP"] = f1_stats_df.loc[tag_name, "TP"]
    df.loc[tag_name, "dual_FP"] = f1_stats_df.loc[tag_name, "FP"]
    df.loc[tag_name, "dual_F1"] = f1_stats_df.loc[tag_name, "F1"]

    f1_stats_df = pd.read_csv(f[N._100_stp], index_col=0)
    df.loc[tag_name, "100_TP"] = f1_stats_df.loc[tag_name, "TP"]
    df.loc[tag_name, "100_FP"] = f1_stats_df.loc[tag_name, "FP"]
    df.loc[tag_name, "100_F1"] = f1_stats_df.loc[tag_name, "F1"]

    f1_stats_df = pd.read_csv(f[N._100_th_stp], index_col=0)
    df.loc[tag_name, "100_th_TP"] = f1_stats_df.loc[tag_name, "TP"]
    df.loc[tag_name, "100_th_FP"] = f1_stats_df.loc[tag_name, "FP"]
    df.loc[tag_name, "100_th_F1"] = f1_stats_df.loc[tag_name, "F1"]

    f1_stats_df = pd.read_csv(f[N._100_cl_stp], index_col=0)
    df.loc[tag_name, "100_cl_TP"] = f1_stats_df.loc[tag_name, "TP"]
    df.loc[tag_name, "100_cl_FP"] = f1_stats_df.loc[tag_name, "FP"]
    df.loc[tag_name, "100_cl_F1"] = f1_stats_df.loc[tag_name, "F1"]


df["percent"] = df["cnt"] / len(files)

# df.to_csv("compare.csv")
print(df)
