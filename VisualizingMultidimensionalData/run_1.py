import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from scipy.sparse import coo_matrix
import numpy as np
from matplotlib import colors as mcolors
from sklearn.cluster import KMeans
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import TruncatedSVD
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates
from wkf.Enums import Category, Country, UseCases, Paths, DataSetType
from wkf.Data import Data


country = Country.BE
category = Category.BEER
use_case = UseCases.UC2
paths = Paths(country=country, category=category, use_case=use_case)
data = Data(paths)
train_set = data.get_data_frame_from_csv(DataSetType.TRAIN)

predictors = [
        "item_description"
]

all_predictors = data.cols_in_train_set.union(data.cols_in_test_set) - data.output_cols_from_metadata - data.cols_in_train_set.symmetric_difference(data.cols_in_test_set)
all_predictors = all_predictors - data.cols_in_train_and_test_but_not_in_metadata
all_predictors = set(['item_description']).union(all_predictors)
predictors = list(all_predictors)

X = train_set[predictors].apply(lambda x: ' '.join(x), axis=1)
y = train_set['if_with_low_calorie_claim']

X_vectorized = CountVectorizer().fit_transform(X)

# reducer = LDA(n_components=3) # fails, when less classes (~2 or 3) are there
# reducer = RandomizedPCA(n_components=3, whiten=False)
# reducer = TruncatedSVD(n_components=3)
reducer = PCA(n_components=3)
km = KMeans(n_clusters=4, init='random', n_init=1, verbose=1)


# X_reduced = pd.DataFrame(reducer.fit_transform(X_vectorized.toarray(), y=y))
aaa = km.fit(X_vectorized.toarray())
labels = aaa.labels_
# X_reduced = pd.DataFrame(aaa.transform(X_vectorized.toarray(), y=y))
X_reduced = pd.DataFrame(reducer.fit_transform(X_vectorized.toarray(), y=y))

print (X_reduced.shape)
print (X_reduced.head())
X_reduced['response'] = y

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys()
response_classes = y.unique()

# for response_class in response_classes:
#     col_index = len(colors) / response_classes.shape[0] * (pd.Index(response_classes).get_loc(response_class) + 1) - 1
#     color = colors[col_index]
#     df_to_plot = X_reduced[X_reduced.response==response_class]
#     plt.scatter(df_to_plot[0], df_to_plot[1], label=response_class, c=color, edgecolor='k')
#
# plt.legend()
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# for response_class in response_classes:
# col_index = len(colors) / response_classes.shape[0] * (pd.Index(response_classes).get_loc(response_class) + 1) - 1
# color = colors[col_index]
df_to_plot = X_reduced
# ax.scatter(df_to_plot[0], df_to_plot[1], df_to_plot[2], c=color, edgecolor='k')
ax.scatter(df_to_plot[0], df_to_plot[1], df_to_plot[2], c=labels.astype(np.float), edgecolor='k')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
