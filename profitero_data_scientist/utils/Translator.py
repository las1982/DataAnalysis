from googletrans import Translator
from profitero_data_scientist.utils.Constants import *
import pandas as pd
import time


class PrepareData:
    jd_reviews_data = None
    reviews_data = None
    product_data = None

    def __init__(self):
        # self.make_data()
        self.start_time = time.time()
        self.translator = Translator()

    def make_data(self):
        self.make_jd_reviews_data()
        self.make_reviews_data()
        self.make_product_data()
        self.write_data_to_csv_files(self.reviews_data, File.review)
        self.write_data_to_csv_files(self.product_data, File.product)

    def make_jd_reviews_data(self):
        self.jd_reviews_data = pd.read_table(File.jd_reviews, encoding='utf-8', quotechar='"', sep=',', dtype='str',
                                             index_col=0)

    def make_reviews_data(self):
        self.reviews_data = pd.DataFrame(self.jd_reviews_data[[Field.review]])
        self.reviews_data.dropna(inplace=True)
        self.reviews_data.drop_duplicates(inplace=True)
        self.reviews_data.reset_index(drop=True, inplace=True)

    def make_product_data(self):
        self.product_data = pd.DataFrame(self.jd_reviews_data[[Field.product_id, Field.product_name]])
        # self.product_data.set_index(Field.product_id, inplace=True)
        self.product_data.dropna(inplace=True)
        self.product_data.drop_duplicates(inplace=True)
        self.product_data.reset_index(drop=True, inplace=True)

    def write_data_to_csv_files(self, data, file):
        data.to_csv(file, encoding='utf-8', index=True, quotechar='"', sep=',')

    def translate_any(self, base_file, trans_file, field_to_trans, field_translated):
        base_df = pd.read_table(base_file, encoding='utf-8', quotechar='"', sep=',', dtype='str', index_col=0)

        try:
            trans_df = pd.read_table(trans_file, encoding='utf-8', quotechar='"', sep=',', dtype='str', index_col=0)
        except:
            trans_df = pd.DataFrame(base_df)
            trans_df[Field.product_name_en] = None
            trans_df.dropna(inplace=True)
            print('no existing file', File.product_en)

        indexes_to_skip = trans_df.index

        for i in base_df.index:
            if i in indexes_to_skip:
                continue
            try:
                text_ch = base_df.loc[i, field_to_trans]
                text_en = self.translate(text_ch)
                for field in base_df.columns.values:
                    trans_df.loc[i, field] = base_df.loc[i, field]
                trans_df.loc[i, field_translated] = text_en
                print(i, 'of', len(base_df.index))
                if i % 100 == 1:
                    trans_df.sort_index(inplace=True)
                    self.write_data_to_csv_files(trans_df, trans_file)
                    print('saved ', i + 1, 'of', len(base_df.index))
            except:
                print('error occured')
                trans_df.sort_index(inplace=True)
                self.write_data_to_csv_files(trans_df, trans_file)
                print('saved ', i + 1, 'of', len(base_df.index))

        self.write_data_to_csv_files(trans_df, trans_file)
        print('saved ', i + 1, 'of', len(base_df.index))

    def translate(self, text):
        if time.time() - self.start_time >= 30 * 60:
            print('waiting and creating new translator')
            time.sleep(1 * 60)
            self.start_time = time.time()
            self.translator = Translator()
        # print('working translator:', self.translator)
        return self.translator.translate(text, dest='en').text


process = PrepareData()
# process.make_data()
process.translate_any(File.review, File.review_en, Field.review, Field.review_en)
# process.translate_any(File.product, File.product_en, Field.product_name, Field.product_name_en)
