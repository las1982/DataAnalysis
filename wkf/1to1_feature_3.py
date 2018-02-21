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
print(df.shape)

print(df[['promotional_activity___type', '_be_loc_var____variety']][df['_be_loc_var____variety'] == 'SCOTCH'])
# print df[['global_outer_packaging', '_be_loc_vari___variteit']].where(df._be_loc_vari___variteit == 'INFERNO')
