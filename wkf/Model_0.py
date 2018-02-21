from wkf.Data import Data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score
import time
from itertools import combinations
from wkf.Enums import Category, Country, UseCases, Paths, DataSetType
import pandas as pd


# class Model:
#
#     def __init__(self, country, category, use_case):
#
#         paths = Paths(country=country, category=category, use_case=use_case)
#         data = Data(paths)
#         self.train = data.get_data_frame_from_csv(DataSetType.TRAIN)
#         self.test = data.get_data_frame_from_csv(DataSetType.TEST)
#         self.additional_predictors = set(['item_description'])
#         # all columns that exist in train_set AND test_set AND metadata, except responses
#         self.all_predictors = data.cols_in_train_set.union(data.cols_in_test_set) - data.output_cols_from_metadata - data.cols_in_train_set.symmetric_difference(data.cols_in_test_set)
#         # all_predictors = all_predictors - set(['european_nan_key', 'item_code', 'european_nan_code', 'external_code', 'standard_code'])
#         self.all_predictors = self.all_predictors - data.cols_in_train_and_test_but_not_in_metadata
#         self.all_predictors = self.additional_predictors.union(self.all_predictors)
#         self.responses = data.output_cols_from_metadata



start_time = time.time()

country = Country.BE
category = Category.BEER
use_case = UseCases.UC2
paths = Paths(country=country, category=category, use_case=use_case)
data = Data(paths)
train_set = data.get_data_frame_from_csv(DataSetType.TRAIN)
test_set = data.get_data_frame_from_csv(DataSetType.TEST)

# all columns that exist in train_set AND test_set AND metadata, except responses
all_predictors = data.cols_in_train_set.union(data.cols_in_test_set) - data.output_cols_from_metadata - data.cols_in_train_set.symmetric_difference(data.cols_in_test_set)
# all_predictors = all_predictors - set(['european_nan_key', 'item_code', 'european_nan_code', 'external_code', 'standard_code'])
all_predictors = all_predictors - data.cols_in_train_and_test_but_not_in_metadata
all_predictors = set(['item_description']).union(all_predictors)
useful_predictors = set([
    'item_description', 'number_in_multipack___actual'
])
responses = data.output_cols_from_metadata
# responses = ['global_brand_extension']
# all_predictors = ['percentage_alcohol_content_by_volume']

comb_size = 3
predictors_lst = list(all_predictors)
# initial_predictors = all_predictors
final_map = {}

for response in responses:

    current_set_of_predictors = all_predictors
    current_best_predictors = []
    tmp_dict_2 = {}

    while True:
        tmp_dict_1 = {}

        for predictor in current_set_of_predictors:

            predictors = [predictor]
            predictors.extend(current_best_predictors)

            X_train = train_set[predictors].apply(lambda x: ' '.join(x), axis=1)
            X_test = test_set[predictors].apply(lambda x: ' '.join(x), axis=1)
            y_train = train_set[response]
            y_test = test_set[response]

            try:
                vect = CountVectorizer().fit(X_train)
                X_train_vectorized = vect.transform(X_train)
                X_test_vectorized = vect.transform(X_test)
            except ValueError:
                print 'error occured, the predictor field has just few words, was working on', predictor, response, ValueError.message
                continue

            # model = MultinomialNB().fit(X_train_vectorized, y_train)
            model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, n_iter=5, random_state=42, n_jobs=-1)
            model.fit(X_train_vectorized, y_train)
            predictions = model.predict(X_test_vectorized)
            f1 = f1_score(y_test, predictions, average='micro')
            tmp_dict_1[tuple(predictors)] = f1
            print 'predictors =', predictors, 'for', 'response =', response, 'F1-score =', f1

        tmp_dict_1_key_with_maximum_f1 = max(tmp_dict_1, key=tmp_dict_1.get)
        current_best_predictors = list(tmp_dict_1_key_with_maximum_f1)
        current_set_of_predictors = current_set_of_predictors - set(current_best_predictors)
        current_max_f1 = tmp_dict_1.get(tmp_dict_1_key_with_maximum_f1, 0.0)

        if len(tmp_dict_2) == 0:
            tmp_dict_2[tmp_dict_1_key_with_maximum_f1] = current_max_f1
            if current_max_f1 == 1:
                break
            else:
                continue
        else:
            if current_max_f1 > tmp_dict_2.get(max(tmp_dict_2, key=tmp_dict_2.get)):
                tmp_dict_2[tmp_dict_1_key_with_maximum_f1] = current_max_f1
                if current_max_f1 == 1:
                    break
                else:
                    continue
            else:
                break

    final_map[response] = {max(tmp_dict_2, key=tmp_dict_2.get): tmp_dict_2.get(max(tmp_dict_2, key=tmp_dict_2.get))}
    print(final_map[response])
    print(final_map)
    # print json.dumps(final_map, default=dumper, indent=2)

print final_map
end_time = time.time()
print 'taken', end_time - start_time, 'sec.'
