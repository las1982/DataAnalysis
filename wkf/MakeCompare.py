import pandas as pd
from os import listdir
from os import walk
import sys

from wkf.Enums import Country, Category, UseCases, Paths

use_case = UseCases.UC2
country = Country.BE
category = Category.BEER
paths = Paths(country=country, category=category, use_case=use_case)

categoryModelsDir = paths.CATEGORY_MODELS_DIR
output_data_file = paths.COMPARE_CSV

df = None
df_columns = []
col_to_report = 'F1'
i = 1
for root, dirs, files in walk(categoryModelsDir, topdown=False):
    print ('next...', i)
    i = i + 1
    if not root.endswith('stats'):
        continue

    for file in files:
        print ('searching...', file)
        if file.endswith('per-tag-stats.csv'):
            print ('working on the', file, 'in the', root)
            modelFolder = root.split('/')[-3:-1][0] + '_' + root.split('/')[-3:-1][1]
            # print (root)
            tmpDF = pd.read_csv(root + '/' + file, index_col=0)
            # print (tmpDF.columns)
            columns = []
            for column in tmpDF.columns:
                # print ('working with column', column)
                columns.append((modelFolder, column))
            # print (columns)
            tmpDF.columns = pd.MultiIndex.from_tuples(columns)
            # tmpDF.set_index('TAG')
            # print (tmpDF.columns)
            # sys.exit(0)
            df_columns.append((modelFolder, col_to_report))
            if df is None:
                df = pd.DataFrame(tmpDF[modelFolder][col_to_report])
                df.columns = pd.MultiIndex.from_tuples(df_columns)
            else:
                dff = pd.DataFrame(tmpDF[modelFolder][col_to_report])
                dff.columns = pd.MultiIndex.from_tuples([(modelFolder, col_to_report)])
                df = df.merge(dff, left_index=True, right_index=True, how='left')
            df.columns = pd.MultiIndex.from_tuples(df_columns)

# df = df.reindex_axis(sorted(df.columns), axis=1) #  - deprecated
df = df.reindex(sorted(df.columns), axis=1)
df.to_csv(output_data_file, encoding='utf-8')
