import pandas as pd
from wkf.Analysis import Analysis
from wkf.Data import Data
from wkf.Enums import Category, Country, UseCases, Paths, DataSetType

use_case = UseCases.UC2
country = Country.DE
category = Category.AIR_CARE
paths = Paths(country=country, category=category, use_case=use_case)
data = Data(paths)

input_cols_for_train_set = data.input_cols_from_metadata.intersection(data.cols_in_train_set)
print (input_cols_for_train_set)
df = data.get_data_frame_from_csv(DataSetType.TRAIN)[['brand_1'] + list(input_cols_for_train_set)]
pd.DataFrame.to_csv(df[df.brand_1 == 'YANKEE (NEWELL BRANDS)'].drop_duplicates(), '/home/alex/work/wf/nielsen/UC2/DE_AIR_CARE/data/brand_1.csv')
