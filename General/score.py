""""
this module defines the score functions we are using
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
import abc
import numpy as np
import sklearn
import pandas as pd
from General.utils import TrainSamples
import scipy.stats as stats

""""""""""""""""""""""""""""""""""""""""""" Class """""""""""""""""""""""""""""""""""""""""""""


class ScoreFunction(abc.ABC):
    """
    An abstract class for ScoreFunction.
    """

    def __init__(self, learning_algorithm: sklearn.base.ClassifierMixin, alpha: int):
        self._learning_algorithm = learning_algorithm
        self._alpha = alpha

    @abc.abstractmethod
    def _execute_function(self, train_samples: TrainSamples, given_features: list[int],
                          new_feature: int, costs_list: list[float]):
        """"
        this function will execute the score function
        """
        ...

    def __call__(self, *args, **kwargs):
        """"
        learning_algorithm default value is None
        alpha default value is 1
        """
        train_samples = args[0]
        given_features = args[1]
        new_feature = args[2]
        costs_list = args[3]

        return self._execute_function(train_samples, given_features, new_feature, costs_list)


class ScoreFunctionA(ScoreFunction):
    """"
    return the feature score according to the theory we explain in the PDF.
    """

    def __init__(self, alpha: int = 1, learning_algorithm: sklearn.base.ClassifierMixin = None):
        super().__init__(learning_algorithm, alpha)

    @staticmethod
    def _get_correlation_to_feature(feature1, feature2, train_samples):
        return abs(np.correlate(feature1, train_samples.samples[feature2])[0])

    def _get_correlation_to_given_features(self, train_samples, new_feature, given_features):
        return np.mean([self._get_correlation_to_feature(train_samples.samples[f], new_feature, train_samples)
                        for f in given_features])

    def _execute_function(self, train_samples: TrainSamples, given_features: list[int],
                          new_feature: int, costs_list: list[float]):
        price = costs_list[new_feature]
        frac = (self._get_correlation_to_feature(train_samples.classes, new_feature, train_samples) /
                self._alpha * self._get_correlation_to_given_features(train_samples, new_feature, given_features))
        return frac / price


class ScoreFunctionB(ScoreFunction):
    """"
    return the feature score according to the theory we explain in the PDF.
    """

    def __init__(self, alpha: int, learning_algorithm: sklearn.base.ClassifierMixin):
        super().__init__(learning_algorithm, alpha)

    def _get_certainty(self, train_samples: TrainSamples, given_features: list[int],
                       new_feature: int):
        """
        return the level of the certainty according to the theory we explain in the PDF.
        :return: the level of the certainty.
        """
        new_features = np.append(given_features, [new_feature])
        data = pd.DataFrame(train_samples.samples)
        self._learning_algorithm.fit(data[new_features], train_samples.classes)
        probabilities = self._learning_algorithm.predict_proba(data[new_features])
        certainty = stats.entropy(probabilities[0], probabilities[1])
        return 1 - certainty

    def _execute_function(self, train_samples: TrainSamples, given_features: list[int],
                          new_feature: int, costs_list: list[float]):
        price = costs_list[new_feature]
        certainty = self._get_certainty(train_samples, given_features, new_feature)
        return certainty / price
