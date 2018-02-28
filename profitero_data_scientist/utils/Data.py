import codecs
from datetime import datetime
import jieba
import pandas as pd
from profitero_data_scientist.utils.Constants import Field, File, Language, Category, CategoryWords


class Data:

    def __init__(self, data_file_csv, language):
        self.reviev_en_file_csv = None
        self.language = language
        self.data_file_csv = data_file_csv

        if language == Language.en:
            self.reviev_en_file_csv = File.review_en
            self.product_en_file_csv = File.product_en
            self.stop_words_file_csv = File.english_stop_words
            self.stop_words = self.get_stop_words()
            self.make_initial_df()
            self.prepare_english_field(Field.review)
            self.prepare_english_field(Field.product_name)
        if language == Language.ch:
            self.stop_words_file_csv = File.chinese_stop_words
            self.stop_words = self.get_stop_words()
            self.make_initial_df()
            self.prepare_chinese_field(Field.review)
            self.prepare_chinese_field(Field.product_name)

        self.prepare_date_field(Field.created_at)

    def make_initial_df(self):

        self.df = pd.read_table(self.data_file_csv, encoding='utf-8', quotechar='"', sep=',', dtype='str')
        self.df.dropna(inplace=True)
        # self.df = self.df.sample(frac=0.01, random_state=10)
        self.df[Field.product_category] = 0

        if self.language == Language.en:
            df_review_en = pd.read_table(self.reviev_en_file_csv, encoding='utf-8', quotechar='"', sep=',', dtype='str',
                                         usecols=[Field.review, Field.review_en])
            self.df = self.df.merge(df_review_en, left_on=Field.review, right_on=Field.review, how='left')
            df_product_en = pd.read_table(self.product_en_file_csv, encoding='utf-8', quotechar='"', sep=',',
                                          dtype='str',
                                          usecols=[Field.product_name, Field.product_name_en])
            self.df = self.df.merge(df_product_en, left_on=Field.product_name, right_on=Field.product_name, how='left')
            self.df[Field.review] = self.df[Field.review_en]
            self.df[Field.product_name] = self.df[Field.product_name_en]
            self.df[Field.product_category] = self.df[Field.product_name].apply(
                lambda product_name: self.categorize_product(product_name))

        self.df = self.df[[
            Field.review_id,
            Field.created_at,
            Field.review,
            Field.product_id,
            Field.product_name,
            Field.product_category,
            Field.stars
        ]]

    def prepare_chinese_field(self, field_name):
        self.df[field_name] = self.df[field_name].apply(
            lambda old_text: self.tokenize_and_delete_stop_words(old_text, self.stop_words))

    def prepare_english_field(self, field_name):
        self.df[field_name] = self.df[field_name].apply(
            lambda old_text: self.delete_stop_words(old_text, self.stop_words))

    def prepare_date_field(self, field_name):
        self.df[field_name] = self.df[field_name].apply(
            lambda date: datetime.strptime(date, '%Y-%m-%d').weekday())

    def tokenize_and_delete_stop_words(self, old_text, stopwords):
        if old_text == '':
            return ''
        new_text = []
        for word in jieba.cut(old_text, cut_all=False, HMM=False):
            if word not in stopwords and word not in [' ', '\n', '\r']:
                new_text.append(word)
        return ' '.join(new_text)

    def delete_stop_words(self, old_text, stopwords):
        if old_text == '':
            return ''
        new_text = []
        for word in old_text.split(' '):
            if word not in stopwords and word not in [' ', '\n', '\r', ',', '.']:
                new_text.append(word)
        return ' '.join(new_text)

    def get_stop_words(self):
        return [line.rstrip() for line in
                codecs.open(self.stop_words_file_csv,
                            "r", encoding="utf-8")]

    def categorize_product(self, product_name):
        if (
                any(word in product_name for word in CategoryWords.words.get(Category.baby_care)) and
                all(word not in product_name for word in CategoryWords.words.get(Category.pet_food))
        ):
            return 1  #Category.baby_care
        elif (
                any(word in product_name for word in CategoryWords.words.get(Category.food)) and
                all(word not in product_name for word in CategoryWords.words.get(Category.pet_food))
        ):
            return 2  #Category.food
        elif (
                any(word in product_name for word in CategoryWords.words.get(Category.snack)) and
                all(word not in product_name for word in CategoryWords.words.get(Category.pet_food))
        ):
            return 3  #Category.snack
        elif (
                any(word in product_name for word in CategoryWords.words.get(Category.mobile))
        ):
            return 4  #Category.mobile
        elif (
                any(word in product_name for word in CategoryWords.words.get(Category.pet_food)) and
                all(word not in product_name for word in CategoryWords.words.get(Category.other))
        ):
            return 5  #Category.pet_food
        else:
            return 6  #Category.other
