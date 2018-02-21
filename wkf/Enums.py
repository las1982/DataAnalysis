class Country:

    def __init__(self):
        pass

    BE = 'BE'
    PT = 'PT'
    US = 'US'
    GB = 'GB'
    IT = 'IT'
    DE = 'DE'
    ES = 'ES'
    NL = 'NL'


class Category:

    def __init__(self):
        pass

    AIR_CARE = 'AIR_CARE'
    BEER = 'BEER'
    WATER = 'WATER'
    SPORT___ENERGY_DRINKS = 'SPORT___ENERGY_DRINKS'
    SHAMPOO_AND_CONDITIONER = 'SHAMPOO_AND_CONDITIONER'
    RUM = 'RUM'
    FLAVOURED_DRINKS___CARBONATED = 'FLAVOURED_DRINKS___CARBONATED'
    LAUNDRY_DETERGENTS = 'LAUNDRY_DETERGENTS'


class UseCases:

    def __init__(self):
        pass

    UC1 = 'UC1'
    UC2 = 'UC2'
    UC3 = 'UC3'
    UC5 = 'UC5'
    UC7 = 'UC7'


class DataSetType:

    def __init__(self):
        pass

    TRAIN = 'train_set'
    TEST = 'test_set'
    TRAIN_TEST = 'test_from_train'   # the test data set was built from the training set
    TRAIN_TRAIN = 'train_from_train'  # the train data set was built from the training set


class Paths:

    def __init__(self, country, category, use_case):
        self.COUNTRY = country
        self.CATEGORY = category
        self.USE_CASE = use_case
        self.TRAIN = DataSetType.TRAIN
        self.TEST = DataSetType.TEST
        self.MAIN_PATH = '/home/alex/work/wf/nielsen/'
        self.META_DATA_CSV = '/home/alex/work/projects/ml-use-cases-alex/nielsen-data-transformation/src/main/resources/metadata/NS_HT_CONFIGURATION.csv'
        self.CATEGORY_MODELS_DIR = self.MAIN_PATH + use_case + '/' + country + '_' + category + '/model/'
        self.COMPARE_CSV = self.CATEGORY_MODELS_DIR + '_'.join([use_case, country, category]) + '_compare.csv'
        self.TRAIN_SET_CSV = self.MAIN_PATH + use_case + '/' + country + '_' + category + '/data/train.csv'
        self.TEST_SET_CSV = self.MAIN_PATH + use_case + '/' + country + '_' + category + '/data/test.csv'
        self.TRAIN_TRAIN_SET_CSV = self.MAIN_PATH + use_case + '/' + country + '_' + category + '/data/train.train.csv'
        self.TRAIN_TEST_SET_CSV = self.MAIN_PATH + use_case + '/' + country + '_' + category + '/data/train.test.csv'
        self.TRAIN_SET_JSON = self.MAIN_PATH + use_case + '/' + country + '_' + category + '/data/train/'
        self.TEST_SET_JSON = self.MAIN_PATH + use_case + '/' + country + '_' + category + '/data/test/'
        self.TEST_OUTPUT_ANALYSIS_FILE = self.MAIN_PATH + use_case + '/' + country + '_' + category + '/data/test_' + category +'_analysis.xlsx'
        self.TRAIN_OUTPUT_ANALYSIS_FILE = self.MAIN_PATH + use_case + '/' + country + '_' + category + '/data/train_' + category +'_analysis.xlsx'