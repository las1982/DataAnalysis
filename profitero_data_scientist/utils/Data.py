import codecs
from datetime import datetime
import jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from profitero_data_scientist.utils import Translator
from profitero_data_scientist.utils.Constants import Field, File, Language, Category, CategoryWords


class Data:

    def __init__(self, data_file_csv, language=Language.en):
        self.data_file_csv = data_file_csv

    def get_original_df(self):
        return pd.read_table(self.data_file_csv, encoding='utf-8', quotechar='"', sep=',', dtype='str')

    def get_translated_df(self):
        df = self.get_original_df()

        Translator.Translate(File.jd_reviews, File.review, Field.review)
        df_review = pd.read_table(File.review, encoding='utf-8', quotechar='"', sep=',', dtype='str', index_col=0)
        df[Field.review] = df[Field.review].apply(
            lambda text: self.translate(text, df_review, Field.review))

        Translator.Translate(File.jd_reviews, File.product_name, Field.product_name)
        df_product = pd.read_table(File.product_name, encoding='utf-8', quotechar='"', sep=',', dtype='str',
                                   index_col=0)
        df[Field.product_name] = df[Field.product_name].apply(
            lambda text: self.translate(text, df_product, Field.product_name))

        return df

    def translate(self, text, dictionary_df, field):
        text = str(text).strip()
        if text in ['nan', '']:
            return np.nan
        return dictionary_df.loc[text, field]


def categorize_date_v1(df, field_name):
    df = df.copy(deep=True)
    df[field_name] = df[field_name].apply(
        lambda date: datetime.strptime(date, '%Y-%m-%d').weekday()
    )
    return df


def categorize_date_v2(df, field_name):
    df = df.copy(deep=True)
    df[field_name] = df[field_name].apply(
        lambda date: datetime.strptime(date, '%Y-%m-%d').month
    )
    return df


def categorize_product(df, field_name):
    df = df.copy(deep=True)
    df[field_name] = df[field_name].apply(
        lambda product_name: categorize(product_name)
    )
    return df


def categorize(product_name):
    if (
            any(word in product_name for word in CategoryWords.words.get(Category.baby_care)) and
            all(word not in product_name for word in CategoryWords.words.get(Category.pet_food))
    ):
        return 1  # Category.baby_care
    elif (
            any(word in product_name for word in CategoryWords.words.get(Category.food)) and
            all(word not in product_name for word in CategoryWords.words.get(Category.pet_food))
    ):
        return 2  # Category.food
    elif (
            any(word in product_name for word in CategoryWords.words.get(Category.snack)) and
            all(word not in product_name for word in CategoryWords.words.get(Category.pet_food))
    ):
        return 3  # Category.snack
    elif (
            any(word in product_name for word in CategoryWords.words.get(Category.mobile))
    ):
        return 4  # Category.mobile
    elif (
            any(word in product_name for word in CategoryWords.words.get(Category.pet_food)) and
            all(word not in product_name for word in CategoryWords.words.get(Category.other))
    ):
        return 5  # Category.pet_food
    else:
        return 6  # Category.other


def make_train_test_files(df):
    train, test, = train_test_split(df)
    train.to_csv(File.train, encoding='utf-8', index=False, quotechar='"', sep=',')
    test.to_csv(File.test, encoding='utf-8', index=False, quotechar='"', sep=',')
