"""
This module contains a partially sophisticated algorithm
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from utils import *
from naive_algorithm import SequenceAlgorithm
import sklearn

""""""""""""""""""""""""""""""""""""""""""" Classes """""""""""""""""""""""""""""""""""""""""""


class MaxVarianceAlgorithm(SequenceAlgorithm):
    """
    A partially sophisticated algorithm that choose to buy the feature with the most variance in each stage.
    """
    def __init__(self, learning_algorithm: sklearn.base.ClassifierMixin):
        """
        Init function.
        :param learning_algorithm: sklearn's classifier. the function saves it and uses it later.
        """
        super().__init__(learning_algorithm)
        self._train_samples = None
        self._features_costs = None
        self._features_by_corr = None

    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        """
        Trains the classifier. the function saves the train samples for the prediction.
        :param train_samples: training dataset contains training data of shape (n_samples, n_features), i.e samples are
        in the rows, and target values of shape (n_samples,). the function saves it.
        :param features_costs: list in length number of features that contains the costs of each feature according to
        indices. in first index you will find the cost of the first feature, etc.
        """
        self._train_samples = train_samples
        self._features_costs = features_costs
        correlations = np.var(train_samples.samples, axis=0)
        args_sort = np.argsort(correlations)
        self._features_by_corr = np.array(args_sort[::-1]).tolist()

    # Private Methods
    def _buy_features(self, given_features: list[int], maximal_cost: float) -> list[int]:
        """
        A method for choosing the supplementary features.
        :param given_features: list of the indices of the chosen features.
        :param maximal_cost: the maximum available cost for buying features.
        :return: the updated given features including all the chosen features.
        """
        new_given_features = given_features
        available_features = list(set(range(self._train_samples.samples.shape[1])) - set(given_features))
        while len(self._features_by_corr) and len(self._features_by_corr) and maximal_cost:
            chosen_feature = self._features_by_corr.pop()
            if chosen_feature not in available_features:
                break
            if self._features_costs[chosen_feature] > maximal_cost:
                break
            maximal_cost -= self._features_costs[chosen_feature]
            new_given_features.append(chosen_feature)
        return new_given_features



