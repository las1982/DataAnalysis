from wkf.Analysis import Analysis
from wkf.Data import Data
from wkf.Enums import Category, Country, UseCases, Paths, DataSetType

use_case = UseCases.UC2
country = Country.DE
category = Category.AIR_CARE
paths = Paths(country=country, category=category, use_case=use_case)
data = Data(paths)

# ************************************************************************
# *********** make xlsx file with browsable analysis *********************
# ************************************************************************
train_analysis = Analysis(data=data, df=data.get_data_frame_from_csv(paths.TRAIN))
train_analysis.make_browsable_analysis(paths.TRAIN_OUTPUT_ANALYSIS_FILE)
# test_analysis = Analysis(data=data, df=data.get_data_frame_from_csv(paths.TEST))
# test_analysis.make_browsable_analysis(paths.TEST_OUTPUT_ANALYSIS_FILE)
# ************************************************************************

# ************************************************************************
# *********** get data.csv file from json paths **************************
# ************************************************************************
# data.make_csv_data_file(DataSetType.TRAIN)
# data.make_csv_data_file(DataSetType.TEST)
# ************************************************************************



