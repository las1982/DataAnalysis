import csv
import json
import re
import sys
import pandas as pd
from os import listdir

from wkf.Enums import Country, Category, UseCases, Paths, DataSetType

csv.field_size_limit(sys.maxsize)


class Data:
    df = None
    input_cols_from_metadata = set()
    output_cols_from_metadata = set()
    cols_in_train_set = set()
    cols_in_test_set = set()
    cols_in_train_and_test_but_not_in_metadata = set()

    def __init__(self, paths):

        self.paths = paths
        self.country = self.paths.COUNTRY
        self.category = self.paths.CATEGORY
        self.use_case = self.paths.USE_CASE
        self.meta_data_csv = self.paths.META_DATA_CSV
        self.train_set_csv = self.paths.TRAIN_SET_CSV
        self.test_set_csv = self.paths.TEST_SET_CSV
        self.train_test_set_csv = self.paths.TRAIN_TEST_SET_CSV
        self.train_train_set_csv = self.paths.TRAIN_TRAIN_SET_CSV
        self.train_set_json = self.paths.TRAIN_SET_JSON
        self.test_set_json = self.paths.TEST_SET_JSON
        self.__get_inp_outp_cols_from_meta()
        self.__get_cols_from_test_set()
        self.__get_cols_from_train_set()
        self.__get_col_descrepency()

    def __normalize_col_names(self):
        self.df = self.df.rename(columns=lambda x: re.sub('[^0-9a-zA-Z]', '_', x).lower())

    def __get_inp_outp_cols_from_meta(self):
        with open(self.meta_data_csv, 'rU') as fin:
            csvin = csv.reader(fin)
            headers = csvin.next()
            for row in csvin:
                if row[headers.index('country')] == self.country \
                        and re.sub('[^A-Z0-9]', '_', row[headers.index('item_category')].upper()) == self.category \
                        and row[headers.index('ht_type')].upper() == self.use_case:
                    input_columns_json = json.loads(row[headers.index('input_columns_json')])
                    for data in input_columns_json:
                        self.input_cols_from_metadata.add(data['id'])
                    output_columns_json = json.loads(row[headers.index('output_columns_json')])
                    for data in output_columns_json:
                        self.output_cols_from_metadata.add(data['id'])
            self.input_cols_from_metadata = self.input_cols_from_metadata - self.output_cols_from_metadata

    def __get_col_descrepency(self):
        self.cols_in_train_and_test_but_not_in_metadata = self.cols_in_test_set.union(self.cols_in_train_set)
        self.cols_in_train_and_test_but_not_in_metadata = self.cols_in_train_and_test_but_not_in_metadata - self.input_cols_from_metadata - self.output_cols_from_metadata

    def __get_cols_from_test_set(self):
        with open(self.test_set_csv, 'rU') as fin:
            csvin = csv.reader(fin)
            self.cols_in_test_set = set(map((lambda x: re.sub('[^0-9a-zA-Z]', '_', x).lower()), csvin.next()))

    def __get_cols_from_train_set(self):
        with open(self.train_set_csv, 'rU') as fin:
            csvin = csv.reader(fin)
            self.cols_in_train_set = set(map((lambda x: re.sub('[^0-9a-zA-Z]', '_', x).lower()), csvin.next()))

    def get_data_frame_from_csv(self, data_set_type):
        print('getting', data_set_type, 'data from csv...')

        if data_set_type == DataSetType.TRAIN:
            data_path = self.train_set_csv
        elif data_set_type == DataSetType.TEST:
            data_path = self.test_set_csv
        elif data_set_type == DataSetType.TRAIN_TRAIN:
            data_path = self.train_train_set_csv
        elif data_set_type == DataSetType.TRAIN_TEST:
            data_path = self.train_test_set_csv

        self.df = pd.read_table(data_path, encoding='iso-8859-1', quotechar='"', sep=',', dtype='str', na_values=["nan"], keep_default_na=False)
        self.__normalize_col_names()
        print('dataframe size:', self.df.size, 'dataframe shape:', self.df.shape)
        return self.df

    def get_data_frame_from_jsons(self, data_set_type):
        print('getting', data_set_type, 'data from jsons...')
        data_path = self.train_set_json if data_set_type == DataSetType.TRAIN else self.test_set_json
        list_json_dict = []
        for json_file in listdir(data_path):
            with open(data_path + json_file, 'r') as f:
                json_dict = json.loads(f.read())
                json_dict['id'] = json_file
                list_json_dict.append(json_dict)
        self.df = pd.DataFrame(list_json_dict)
        self.df.set_index('id', inplace=True)
        self.__normalize_col_names()
        print('dataframe size:', self.df.size, 'dataframe shape:', self.df.shape)
        return self.df

    def make_csv_data_file(self, data_set_type):
        print('making', data_set_type, 'data to csv file...')
        data_csv = self.train_set_csv if data_set_type == DataSetType.TRAIN else self.test_set_csv
        df = self.get_data_frame_from_jsons(data_set_type)
        df.to_csv(data_csv, encoding='utf-8', index=True, quotechar='"', sep=',')


def test():
    country = Country.US
    category = Category.SHAMPOO_AND_CONDITIONER
    use_case = UseCases.UC3
    paths = Paths(country=country, category=category, use_case=use_case)
    data = Data(paths)
    train_set_from_jsons = data.get_data_frame_from_jsons(paths.TRAIN)
    test_set_from_jsons = data.get_data_frame_from_jsons(paths.TEST)
    train_set_from_csv = data.get_data_frame_from_csv(paths.TRAIN)
    test_set_from_csv = data.get_data_frame_from_csv(paths.TEST)

    input_cols_from_metadata = data.input_cols_from_metadata
    output_cols_from_metadata = data.output_cols_from_metadata

    cols_in_train_set = data.cols_in_train_set
    cols_in_test_set = data.cols_in_test_set

    cols_in_train_but_not_in_test = cols_in_train_set - cols_in_test_set
    cols_in_test_but_not_in_train = cols_in_test_set - cols_in_train_set
    train_test_cols_symmetric_difference = cols_in_train_set.symmetric_difference(cols_in_test_set)

    cols_in_datasets_but_not_in_metadata = data.cols_in_train_and_test_but_not_in_metadata
    print('done')

# test()
