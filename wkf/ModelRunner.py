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
from wkf.Data import Data
from wkf.Enums import Category, Country, UseCases, Paths, DataSetType


start_time = time.time()

country = Country.BE
category = Category.BEER
use_case = UseCases.UC2
paths = Paths(country=country, category=category, use_case=use_case)
data = Data(paths)
train = data.get_data_frame_from_csv(DataSetType.TRAIN)
test = data.get_data_frame_from_csv(DataSetType.TEST)

all_predictors = data.cols_in_train_set.union(data.cols_in_test_set) - data.output_cols_from_metadata - data.cols_in_train_set.symmetric_difference(data.cols_in_test_set)
# all_predictors = all_predictors - set(['european_nan_key', 'item_code', 'european_nan_code', 'external_code', 'standard_code'])
all_predictors = all_predictors - data.cols_in_train_and_test_but_not_in_metadata
all_predictors = set(['item_description']).union(all_predictors)
useful_predictors = set([
    'item_description', 'number_in_multipack___actual'
])
responses = data.output_cols_from_metadata
responses = ['global_number_in_multipack___actual']
# all_predictors = ['percentage_alcohol_content_by_volume']

comb_size = 3

for response in responses:
    for combination in combinations(all_predictors, comb_size):
        if len(set(combination) - useful_predictors) != comb_size - len(useful_predictors) \
                and len(useful_predictors) != 0:
            continue
        predictors = list(combination)
        try:
            X_train = train[predictors].apply(lambda x: ' '.join(x), axis=1)
            X_test = test[predictors].apply(lambda x: ' '.join(x), axis=1)
        except ValueError:
            print 'error occured, there is no such a field in a data set', predictors, response, ValueError.message
            continue
        y_train = train[response]
        y_test = test[response]

        try:
            vect = CountVectorizer().fit(X_train)
            # vect = TfidfVectorizer().fit(X_train)
        except ValueError:
            print 'error occured, was working on', predictors, response, ValueError.message
            continue

        # transform the documents in the training data to a document-term matrix
        X_train_vectorized = vect.transform(X_train)
        X_test_vectorized = vect.transform(X_test)

        # Train the model

        # model = MultinomialNB().fit(X_train_vectorized, y_train)
        model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, n_iter=5, random_state=42, n_jobs=-1)
        # model = LogisticRegression(multi_class='multinomial', solver='newton-cg')
        # model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3),
            }
        # model = GridSearchCV(model, parameters, n_jobs=1)
        # model = GridSearchCV(model, parameters)

        # Fit the model
        model.fit(X_train_vectorized, y_train)

        # Predict the transformed test_set documents
        predictions = model.predict(X_test_vectorized)

        # for i in range(len(X_test)):
            # print X_test[i] + '\t' + y_test[i] + '\t' + predictions[i]

        # print('confusion_matrix:\n', confusion_matrix(y_test, predictions))

        print 'predictors =', predictors, 'for', 'response =', response, 'F1-score =', f1_score(y_test, predictions, average='micro')

end_time = time.time()
print 'taken', end_time - start_time, 'sec.'
