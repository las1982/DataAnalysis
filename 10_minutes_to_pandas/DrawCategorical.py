import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

np.random.seed(sum(map(ord, "categorical")))
titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

sns.stripplot(x="day", y="total_bill", data=tips)
plt.show()
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
plt.show()
sns.swarmplot(x="day", y="total_bill", data=tips)
plt.show()
sns.swarmplot(x="day", y="total_bill", hue="sex", data=tips)
plt.show()
sns.swarmplot(x="size", y="total_bill", data=tips)
plt.show()
sns.swarmplot(x="total_bill", y="day", hue="time", data=tips)
plt.show()
sns.boxplot(x="day", y="total_bill", hue="time", data=tips)
plt.show()
tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
sns.boxplot(x="day", y="total_bill", hue="weekend", data=tips)
plt.show()
sns.violinplot(x="total_bill", y="day", hue="time", data=tips)
plt.show()
sns.violinplot(x="total_bill", y="day", hue="time", data=tips,
               bw=.1, scale="count", scale_hue=False)
plt.show()
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True)
plt.show()
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips,
               split=True, inner="stick", palette="Set3")
plt.show()
sns.violinplot(x="day", y="total_bill", data=tips, inner=None)
sns.swarmplot(x="day", y="total_bill", data=tips, color="w", alpha=.5)
plt.show()
sns.barplot(x="sex", y="survived", hue="class", data=titanic)
plt.show()
sns.countplot(x="deck", data=titanic, palette="Greens_d")
plt.show()
sns.countplot(y="deck", hue="class", data=titanic, palette="Greens_d")
plt.show()
sns.pointplot(x="sex", y="survived", hue="class", data=titanic)
plt.show()
sns.pointplot(x="class", y="survived", hue="sex", data=titanic,
              palette={"male": "g", "female": "m"},
              markers=["^", "o"], linestyles=["-", "--"])
plt.show()
sns.boxplot(data=iris, orient="h")
plt.show()
sns.violinplot(x=iris.species, y=iris.sepal_length)
plt.show()
f, ax = plt.subplots(figsize=(7, 3))
sns.countplot(y="deck", data=titanic, color="c")
plt.show()
sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips)
plt.show()
sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips, kind="bar")
plt.show()
sns.factorplot(x="day", y="total_bill", hue="smoker",
               col="time", data=tips, kind="swarm")
plt.show()
sns.factorplot(x="time", y="total_bill", hue="smoker",
               col="day", data=tips, kind="box", size=4, aspect=.5)
plt.show()
g = sns.PairGrid(tips,
                 x_vars=["smoker", "time", "sex"],
                 y_vars=["total_bill", "tip"],
                 aspect=.75, size=3.5)
g.map(sns.violinplot, palette="pastel")
plt.show()












