import sys

import pandas as pd
import matplotlib.pyplot as plt
from wkf.CHUBBINT.Const import File, Column
from sklearn.model_selection import train_test_split


class InfoPrinter:

    def __init__(self, df):
        self.df = df
        self.df_to_plot = df.groupby(Column.category)[Column.desc_tagged].count().sort_values(ascending=False)

    def print(self):
        print('{0:25} ==> {1}'.format("df shape", self.df.shape))
        print('{0:25} ==> {1}'.format("descr shape", self.df[Column.desc_tagged].shape))
        print('{0:25} ==> {1}'.format("descr drop duplic", self.df[Column.desc_tagged].drop_duplicates().shape))
        print('{0:25} ==> {1}'.format("descr drop dupl, drop na",
                                      self.df[Column.desc_tagged].drop_duplicates().dropna().shape))
        print('{0:25} ==> {1}'.format("categ shape", self.df[Column.category].shape))
        print('{0:25} ==> {1}'.format("categ drop dupl", self.df[Column.category].drop_duplicates().shape))
        print(
            '{0:25} ==> {1}'.format("categ drop dup, drop na",
                                    self.df[Column.category].drop_duplicates().dropna().shape))
        print()

    def print_distr(self):
        print(self.df_to_plot)
        print()

    def plot_distr(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = pd.Series(self.df_to_plot, index=range(self.df_to_plot.shape[0])).index
        x_lab = self.df_to_plot.index
        y = self.df_to_plot.values
        rects1 = ax.bar(x, y)
        # ax.set_xticklabels(x_lab)
        plt.xticks(x, x_lab, rotation=90, wrap=True)
        # plt.xticks(x, x_lab, rotation=90)
        plt.show()


df = pd.read_table(File.data_body, encoding='utf-8', quotechar='"', sep=',', dtype='str')

cat = [
    "Abdomen Including Groin",
    "Upper Arm (inc. Clavicle & Scapula)",
    "Chest (inc. Ribs, Sternum, and Soft Tissue)",
    "Multiple Upper Extremities",
    "Multiple Head Injury",
    "Upper Leg",
    "Insufficient Info to Properly Identify - Unclassified",
    "Wrist(s) and Hand(s)",
    "Thumb",
    "Multiple Body Parts",
    "Lower Leg",
    "Elbow",
    "No Physical Injury",
    "Knee",
    "Multiple Lower Extremities",
    "Eye(s)",
    "Lower Arm",
    "Body Systems and Multiple Body Systems",
    "Wrist"
]
# df = df[df[Column.category].isin(cat)]

antology = pd.read_table("anthology.csv", encoding='utf-8', quotechar='"', sep=',', dtype='str', index_col=0)

df[Column.category + "_l2"] = df[Column.category].apply(lambda x: antology.loc[x, "l2"])


infoPrinter = InfoPrinter(df)
infoPrinter.print()
infoPrinter.print_distr()

df_train, df_test = train_test_split(
    df,
    shuffle=True,
    stratify=df[Column.category],
    # random_state=42,
    test_size=0.25)
df_train = df_train.drop_duplicates().dropna()
df_test = df_test.drop_duplicates().dropna()

infoPrinter = InfoPrinter(df_train)
infoPrinter.print()
# infoPrinter.print_distr()
# infoPrinter.plot_distr()
df_train.to_csv(File.train, encoding='utf-8', index=True, quotechar='"', sep=',')

infoPrinter = InfoPrinter(df_test)
infoPrinter.print()
# infoPrinter.print_distr()
# infoPrinter.plot_distr()
df_test.to_csv(File.test, encoding='utf-8', index=True, quotechar='"', sep=',')
