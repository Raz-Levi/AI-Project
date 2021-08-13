"""
This module defines an abstract algorithm class.
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
import abc
from utils import *

""""""""""""""""""""""""""""""""""""""""""" Class """""""""""""""""""""""""""""""""""""""""""


class LearningAlgorithm(abc.ABC):
    """
    An abstract class for LearningAlgorithm.
    """
    def __init__(self):
        self._total_features_num = None

    ### Public Methods ###
    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        """
        Fits the classifier. the function saves the number of the features in training dataset and calls to an abstract
        fit method.
        :param train_samples: training dataset contains training data of shape (n_samples, n_features), i.e samples are
        in the rows, and target values of shape (n_samples,).
        :param features_costs: list in length number of features that contains the costs of each feature according to
        indices. in first index you will find the cost of the first feature, etc.
        """
        self._total_features_num = train_samples.samples.shape[1]
        self._fit(train_samples, features_costs)

    @abc.abstractmethod
    def predict(self, samples: TestSamples, given_features: list[int], maximal_cost: float) -> np.array:
        """
        An abstract method which predicts the class labels for the provided data.
        :param samples: test samples of shape (n_samples, n_features), i.e samples are in the rows.
        :param given_features: list of the indices of the chosen features.
        :param maximal_cost: the maximum available cost for buying features.
        :return: np.array of shape (n_samples,) contains the class labels for each data sample.
        """
        ...

    ### Private Methods ###
    @abc.abstractmethod
    def _fit(self, train_samples: TrainSamples, features_costs: list[float]):
        """
        An abstract method for fit method which implements the training process of the algorithm. it is called by the
        fit method.
        :param train_samples: training dataset contains training data of shape (n_samples, n_features), i.e samples are
        in the rows, and target values of shape (n_samples,).
        :param features_costs: list in length number of features that contains the costs of each feature according to
        indices. in first index you will find the cost of the first feature, etc.
        """
        ...
