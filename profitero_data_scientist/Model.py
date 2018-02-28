import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

from profitero_data_scientist.utils.Constants import File, Models


class Model:

    def __init__(self, model_name):
        self.model = None
        self.predictions = None
        self.model_name = model_name
        self.model_file = '/'.join([File.models_dir, self.model_name])
        if model_name == Models.model_1_sgd:
            self.model = SGDClassifier(
                loss="hinge",
                penalty='l2',
                alpha=0.00001,
                l1_ratio=0.15,
                fit_intercept=True,
                max_iter=100,
                tol=None,
                shuffle=True,
                verbose=0,
                # epsilon=None,
                n_jobs=-1,
                random_state=None,
                learning_rate="optimal",
                eta0=0.0,
                power_t=0.5,
                class_weight='balanced',
                warm_start=False,
                average=False,
                n_iter=None
            )

        if model_name == Models.model_2_mpl:
            self.model = MLPClassifier(
                hidden_layer_sizes=(50, 25, 10),
                activation="relu",
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate="constant",
                learning_rate_init=0.001,
                power_t=0.5,
                max_iter=6,
                shuffle=True,
                random_state=None,
                tol=1e-4,
                verbose=True,
                warm_start=False,
                momentum=0.9,
                nesterovs_momentum=True,
                early_stopping=False,
                validation_fraction=0.1,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8
            )

    def train(self, X, Y):
        self.model.fit(X, Y)
        self.save()

    def save(self):
        pickle.dump(self.model, open(self.model_file, 'wb'))

    def load(self):
        self.model = pickle.load(open(self.model_file, 'rb'))

    def extract(self, X, Y):
        self.load()
        self.predictions = self.model.predict(X)
        print(classification_report(Y, self.predictions))
