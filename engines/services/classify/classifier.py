import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from engines.services.data_collection.utils import OTHERS
from engines.services.representation import (
    TFIDFRepresentation,
    BertRepresentation,
    FasttextRepresentation, BaseRepresentation, check_mkdir
)

_representations = {
    'tf-idf': TFIDFRepresentation,
    'bert': BertRepresentation,
    'fasttext': FasttextRepresentation
}


class _BaseClassifier:
    def __init__(
            self, data=None,
            load_clf: bool = False,
            method: str = 'tf-idf',
            split_test_size: float = 0.1,
            split_random_state: float = 1,
            classifier_root: str = 'models',
            classifier_folder: str = 'classifiers',
            representation: BaseRepresentation = None,
            **repr_kwargs
    ):
        check_mkdir(classifier_root)
        self.path = os.path.join(classifier_root, classifier_folder)
        check_mkdir(self.path)
        self.representation = _representations[method](data=data, **repr_kwargs) \
            if not representation else representation
        self.classifier = None if not load_clf else self.load_model(self.model_path)

        if data:
            self.X, self.y = self._getXy(data)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y,
                test_size=split_test_size,
                random_state=split_random_state
            )
            self.y_predicted = None

    @property
    def model_path(self):
        raise NotImplementedError

    def _getXy(self, data):
        X = self.representation.represent().values
        y = np.array([
            doc['category'] if doc['category'] else OTHERS
            for doc in data
        ])
        return X, y

    @classmethod
    def load_model(cls, path: str):
        return pickle.load(open(path, 'rb'))

    def save_model(self, path: str):
        pickle.dump(self.classifier, open(path, 'wb'))

    def f1_score(self):
        return f1_score(self.y_test, self.y_predicted, average='macro')

    def accuracy(self):
        return accuracy_score(self.y_test, self.y_predicted)

    def confusion_matrix(self, plot: bool = True):
        if plot:
            plot_confusion_matrix(self.classifier, self.X_test, self.y_test)
        return confusion_matrix(self.y_test, self.y_predicted, labels=self.y)

    def build(self, save: bool = True):
        raise NotImplementedError

    def classify(self, query):
        vector = self.representation.embed(query)
        y_pred = self.classifier.predict([vector])
        return y_pred[0]


class LogisticRegressionClassifier(_BaseClassifier):
    _FILE = 'logistic_regr.pkl'

    @property
    def model_path(self):
        return os.path.join(self.path, self._FILE)

    def build(self, save: bool = True, random_state: float = 0):
        self.classifier = LogisticRegression(
            random_state=random_state,
            max_iter=300
        ).fit(
            self.X_train, self.y_train
        )
        self.y_predicted = self.classifier.predict(self.X_test)
        if save:
            self.save_model(self.model_path)
        return self.classifier


class NaiveBayesClassifier(_BaseClassifier):
    _FILE = 'naive_bayes.pkl'

    @property
    def model_path(self):
        return os.path.join(self.path, self._FILE)

    def build(self, save: bool = True):
        self.classifier = GaussianNB()
        self.classifier.fit(self.X_train, self.y_train)
        self.y_predicted = self.classifier.predict(self.X_test)
        if save:
            self.save_model(self.model_path)
        return self.classifier
