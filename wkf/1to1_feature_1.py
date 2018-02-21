import pandas as pd
from Data import fields
import re
import operator
from itertools import combinations


predictors = fields.get('L2G_BE_BEER').get('input')
predictors.extend([
    "item_description",
    "global_module",
    "global_product_group",
    "item_has_images",
    "item_creation_date",
    "shared_item_indication",
    "historical_cross_codings"
])

responces = fields.get('L2G_BE_BEER').get('output')
predictors.sort()


def get_ngram_list(input_list=list(), n=2):
    ngram_list = []
    for i in range(len(input_list) - n + 1):
        ngram_list.append(input_list[i:i + n])
    return ngram_list


csv_file = '/media/alex/data/alex/work/wkf/nielsen/UC2/L2G_BE_BEER/data/train_set.csv'
df = pd.read_table(csv_file, ',', quotechar='"')
df = df.rename(columns=lambda x: re.sub('[^0-9a-zA-Z]', '_', x).lower())
print df.shape
df = df[df.global_collection_claim != 'NO CLAIM']
print df.shape

for responce in ['global_collection_claim']:
    map = {}
    column_ngrams = combinations(predictors, 2)
    i = 1
    for column_ngram in column_ngrams:
        print 'working on', i
        i = i + 1
        responce_unique_count = pd.DataFrame(df[[responce]]).drop_duplicates().size

        # column_ngram = column_ngram + (responce, )
        tmp_df = pd.DataFrame(df[list(column_ngram)]).drop_duplicates()
        if tmp_df.shape[0] == responce_unique_count:
            column_ngram = column_ngram + (responce, )
            tmp_df1 = pd.DataFrame(df[list(column_ngram)]).drop_duplicates()
            if tmp_df1.shape[0] == responce_unique_count:
                map[column_ngram] = tmp_df1.shape[0]
    # sorted_x = sorted(map.items(), key=operator.itemgetter(1))
    sorted_x = sorted(map.items(), key=lambda x: x[1], reverse=False)
    print sorted_x[:10]


