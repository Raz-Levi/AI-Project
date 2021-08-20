"""
This module defines an abstract algorithm class.
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
import abc
from General.utils import *

""""""""""""""""""""""""""""""""""""""""""" Class """""""""""""""""""""""""""""""""""""""""""


class LearningAlgorithm(abc.ABC):
    """
    An abstract class for LearningAlgorithm.
    """
    @abc.abstractmethod
    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        """
        An abstract method for training the classifier.
        :param train_samples: training dataset contains training data of shape (n_samples, n_features), i.e samples are
        in the rows, and target values of shape (n_samples,).
        :param features_costs: list in length number of features that contains the costs of each feature according to
        indices. in first index you will find the cost of the first feature, etc.
        """
        ...

    @abc.abstractmethod
    def predict(self, samples: TestSamples, given_features: list[int], maximal_cost: float) -> Classes:
        """
        An abstract method for predicting the class labels for the provided data.
        :param samples: test samples of shape (n_samples, n_features), i.e samples are in the rows.
        :param given_features: list of the indices of the chosen features.
        :param maximal_cost: the maximum available cost for buying features.
        :return: Classes of shape (n_samples,) contains the class labels for each data sample.
        """
        ...


class SequenceAlgorithm(LearningAlgorithm):
    """
    An abstract naive algorithm that chooses simply features to add to the given features according to the given budget.
    """
    # Public Methods
    def __init__(self, learning_algorithm: sklearn.base.ClassifierMixin):
        """
        Init function.
        :param learning_algorithm: sklearn's classifier. the function saves it and uses it later.
        """
        super().__init__()
        self._train_samples = None
        self._features_costs = []
        self._learning_algorithm = learning_algorithm

    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        """
        Trains the classifier. the function saves the train samples and the features costs for the prediction.
        :param train_samples: training dataset contains training data of shape (n_samples, n_features), i.e samples are
        in the rows, and target values of shape (n_samples,). the function saves it.
        :param features_costs: list in length number of features that contains the costs of each feature according to
        indices. in first index you will find the cost of the first feature, etc. the function saves it.
        """
        self._train_samples = train_samples
        self._features_costs = features_costs

    def predict(self, samples: TestSamples, given_features: list[int], maximal_cost: float) -> Classes:
        """
        Predicts the class labels for the provided data. the function chooses simply features to add to the given
        features according to the given budget and runs the fit and predict function of the given sklearn's classifier
        on the given test samples filtering the given features.
        :param samples: test samples of shape (n_samples, n_features), i.e samples are in the rows.
        :param given_features: list of the indices of the chosen features.
        :param maximal_cost: the maximum available cost for buying features.
        :return: Classes of shape (n_samples,) contains the class labels for each data sample.
        """
        given_features = self._buy_features(given_features, maximal_cost)
        self._learning_algorithm.fit(self._train_samples.samples[:, given_features], self._train_samples.classes)
        return self._learning_algorithm.predict(samples[:, given_features])

    # Private Methods
    @abc.abstractmethod
    def _buy_features(self, given_features: list[int], maximal_cost: float) -> list[int]:
        """
        An abstract method for choosing the supplementary features.
        :param given_features: list of the indices of the chosen features.
        :param maximal_cost: the maximum available cost for buying features.
        :return: the updated given features including all the chosen features.
        """
        ...

