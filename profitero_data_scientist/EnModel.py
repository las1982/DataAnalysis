import numpy as np
import pandas as pd
from scipy.sparse import hstack
from profitero_data_scientist.utils.Model import Model
from profitero_data_scientist.utils.Vectorizer import Vectorizer
from profitero_data_scientist.utils import Data
from profitero_data_scientist.utils.Constants import File, Field, Models, Vectorizers


def prepare(X):
    X_vectorized = Vectorizer(Vectorizers.en_revi_1_5_ngram_count).transform(X[Field.review])
    # tmp = Vectorizer(Vectorizers.en_prod_1_2_ngram_count).transform(X[Field.product_name])
    # X_vectorized = hstack((X_vectorized, tmp))

    a = Data.categorize_date_v1(X, Field.created_at)
    X_vectorized = hstack((X_vectorized, np.array(a[Field.created_at])[:, None]))

    b = Data.categorize_date_v2(X, Field.created_at)
    X_vectorized = hstack((X_vectorized, np.array(b[Field.created_at])[:, None]))

    # c = Data.categorize_product(X, Field.product_name)
    # X_vectorized = hstack((X_vectorized, np.array(c[Field.product_name])[:, None]))

    return X, X_vectorized


def make_data():
    data = Data.Data(File.jd_reviews)
    df = data.get_original_df()
    Data.make_train_test_files(df)
    # df = df.sample(frac=0.01, random_state=10)


def train(train_file=None):
    print('training model')
    if train_file is None:
        make_data()
        train_file = File.train

    train = Data.Data(train_file).get_translated_df()
    # train = pd.read_table(train_file, encoding='utf-8', quotechar='"', sep=',', dtype='str')
    train.dropna(inplace=True)

    X_train = train[[Field.review, Field.created_at, Field.product_name]]
    Y_train = train[[Field.stars]]

    X_train, X_train_vectorized = prepare(X_train)
    print('X_train:', X_train.shape, 'Y_train:', Y_train.shape)
    print('X_train_vectorized:', X_train_vectorized.shape)

    model = Model(Models.model_en)
    model.train(X_train_vectorized, Y_train.values.ravel())


def extract(test_file):
    print('extracting from model')

    test = Data.Data(test_file).get_translated_df()

    id_before = test[Field.review_id]
    test.dropna(inplace=True)
    id_after = test[Field.review_id]
    id_for_na = list(set(id_before.values.tolist()) - set(id_after.values.tolist()))

    X_test = test[[Field.review, Field.created_at, Field.product_name]]
    Y_test = test[[Field.stars]]

    X_test, X_test_vectorized = prepare(X_test)
    print('X_test:', X_test.shape, 'Y_test:', Y_test.shape)
    print('X_test_vectorized', X_test_vectorized.shape)

    model = Model(Models.model_en)
    model.extract(X_test_vectorized, Y_test)

    output = pd.DataFrame({Field.review_id: test[Field.review_id],
                           Field.stars: model.predictions
                           })
    na = pd.DataFrame({Field.review_id: pd.Series(id_for_na),
                       Field.stars: ''
                       })
    output = output.append(na, ignore_index=True)
    output.to_csv(File.output, encoding='utf-8', index=False, quotechar='"', sep=',')


# train()
extract(File.test)
