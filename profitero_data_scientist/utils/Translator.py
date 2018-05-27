from googletrans import Translator
from profitero_data_scientist.utils.Constants import *
import pandas as pd
import time


class Translate:

    def __init__(self, src_file, dst_file, field, language='en'):
        self.translator = Translator()
        self.start_time = time.time()

        self.src_file = src_file
        self.dst_file = dst_file
        self.field = field
        self.language = language
        self.src_texts = None
        self.processed_texts = None
        self.dst_df = None

        self.load()
        self.prepare_texts_to_translate()
        self.translate()

    def load(self):
        try:
            self.dst_df = pd.read_table(self.dst_file, encoding='utf-8', quotechar='"', sep=',', dtype='str',
                                        index_col=0)
        except:
            self.dst_df = pd.DataFrame(columns=[self.field])
            self.save()

    def save(self):
        self.dst_df.sort_index(inplace=True)
        self.dst_df.to_csv(self.dst_file, encoding='utf-8', index=True, quotechar='"', sep=',')
        print('saved ', self.dst_df.shape[0], 'of', self.docs_to_translate)

    def prepare_texts_to_translate(self):
        self.src_texts = pd.read_table(self.src_file, encoding='utf-8', quotechar='"', sep=',', dtype='str',
                                       usecols=[self.field])
        self.src_texts = self.src_texts[self.field]
        self.src_texts.dropna(inplace=True)
        self.src_texts = self.src_texts.apply(lambda text: text.strip())
        self.src_texts.drop_duplicates(inplace=True)
        self.docs_to_translate = self.src_texts.size
        self.src_texts = self.src_texts.tolist()
        self.src_texts = sorted(list(set(self.src_texts) - set(self.dst_df.index.values.tolist())))

    def translate(self):
        cnt = 0
        print('translating', self.field)
        for src_text in self.src_texts:
            self.dst_df.loc[src_text, self.field] = self.translate_text(src_text)
            self.save()
            cnt += 1
        if cnt == 0:
            print('nothing to translate for', self.field)
        else:
            print('translated for', self.field, cnt, 'doc')

    def translate_text(self, text):
        wait_after = 30 * 60
        waiting_time = 1 * 60
        if time.time() - self.start_time >= wait_after:
            print('waiting', waiting_time, 'sec and creating new translator')
            time.sleep(waiting_time)
            self.start_time = time.time()
            self.translator = Translator()
        return self.translator.translate(text, dest=self.language).text


# Translate(File.jd_reviews, File.product_name, Field.product_name)
# Translate(File.jd_reviews, File.review, Field.review)
