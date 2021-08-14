"""
This module defines an abstract algorithm class.
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from utils import *

""""""""""""""""""""""""""""""""""""""""""" Class """""""""""""""""""""""""""""""""""""""""""


class LearningAlgorithm(abc.ABC):
    """
    An abstract class for LearningAlgorithm.
    """
    @abc.abstractmethod
    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        """
        An abstract method for training the classifier. the function saves the number of the features in training dataset and calls to an abstract
        fit method.
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
        :return: np.array of shape (n_samples,) contains the class labels for each data sample.
        """
        ...
