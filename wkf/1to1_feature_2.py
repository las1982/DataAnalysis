import pandas as pd
from Data import fields
import re
import operator
from itertools import combinations


predictors = fields.get('L2G_BE_BEER').get('input')
predictors.extend([
    # "item_description",
    "global_module",
    "global_product_group",
    "item_has_images",
    # "item_creation_date",
    "shared_item_indication",
    "historical_cross_codings"
])

responces = fields.get('L2G_BE_BEER').get('output')
predictors.sort()

csv_file = '/media/alex/data/alex/work/wkf/nielsen/UC2/L2G_BE_BEER/data/train_set.csv'
df = pd.read_table(csv_file, ',', quotechar='"')
df = df.rename(columns=lambda x: re.sub('[^0-9a-zA-Z]', '_', x).lower())
# print df.shape
# df = df[df.global_collection_claim != 'NO CLAIM']
print df.shape

comb_size = 1
res_df_cols = ['1', '2', '3', '4']
# for i in range(comb_size):
#     res_df_cols.append('col_' + json_str(i) + '_name')
#     res_df_cols.append('col_' + json_str(i) + '_val')
# res_df_cols.append('target_name')
# res_df_cols.append('target_value')
res_df = pd.DataFrame(columns=res_df_cols)

for responce in ['global_collection_claim']:
    class_vals = df.global_collection_claim.drop_duplicates()
    class_vals = class_vals[class_vals != 'NO CLAIM']
    column_ngrams = combinations(predictors, comb_size)
    map = {}
    i = 1
    for column_ngram in column_ngrams:
        print 'working on', i, 'map size =', len(map)
        i = i + 1
        for class_val in class_vals:
            tmp_df_1 = df[list(column_ngram) + [responce]].where(df[responce] == class_val).drop_duplicates().dropna(axis=0, how='all')
            vals = tmp_df_1[list(column_ngram)]
            tmp_df_2 = df[list(column_ngram) + [responce]].drop_duplicates().dropna(axis=0, how='all')
            result = pd.merge(tmp_df_1, tmp_df_2, how='left', on=list(column_ngram))
            if tmp_df_1.shape[0] == result.shape[0]:
                tmp_df_1['col'] = tmp_df_1.columns[0]
                tmp_df_1['tag'] = tmp_df_1.columns[-2]
                tmp_df_1.columns = res_df_cols
                res_df = res_df.append(tmp_df_1)
                # map[column_ngram + (json_str(vals),)] = class_val

    # sorted_x = sorted(map.items(), key=operator.itemgetter(1))
    # sorted_x = sorted(map.items(), key=lambda x: x[1], reverse=False)
    print res_df.describe()


