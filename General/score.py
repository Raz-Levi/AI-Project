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


class ScoreFunction:
    """
    An abstract class for ScoreFunction.
    """

    def __init__(self, learning_algorithm: sklearn.base.ClassifierMixin, alpha: int):
        """"
        :param learning_algorithm : a base algorithm in which the function rely on
        :param alpha : represent the ratio between function's params
        """
        self._learning_algorithm = learning_algorithm
        self._alpha = alpha

    @abc.abstractmethod
    def _execute_function(self):
        """"
        this function will execute the score function
        """
        ...

    def __call__(self, *args, **kwargs):
        return self._execute_function()


class ScoreFunctionA(ScoreFunction):
    """"
    return the feature score according to the theory we explain in the PDF.
    """
    def __init__(self, train_samples: TrainSamples, given_features: list[int], new_feature: int,
                 costs_list: list[float], learning_algorithm: sklearn.base.ClassifierMixin = None, alpha: int = 1):
        super().__init__(learning_algorithm, alpha)
        self._train_samples = train_samples
        self._given_features = given_features
        self._new_feature = new_feature
        self._costs_list = costs_list

    def __get_correlation_to_feature(self, feature1, feature2):
        y = feature1
        x = self._train_samples.samples[feature2]
        z = np.correlate(feature1, self._train_samples.samples[feature2])
        return abs(np.correlate(feature1, self._train_samples.samples[feature2])[0])

    def __get_correlation_to_given_features(self):
        return np.mean([self.__get_correlation_to_feature(self._train_samples.samples[f], self._new_feature)
                        for f in self._given_features])

    def _execute_function(self):
        price = self._costs_list[self._new_feature]
        frac = (self.__get_correlation_to_feature(self._train_samples.classes, self._new_feature) /
                self._alpha * self.__get_correlation_to_given_features())
        return frac / price


class ScoreFunctionB(ScoreFunction):
    """"
    return the feature score according to the theory we explain in the PDF.
    """
    def __init__(self, train_samples: TrainSamples, given_features: list[int], new_feature: int,
                 costs_list: list[float], learning_algorithm: sklearn.base.ClassifierMixin, alpha: int = 1):
        super().__init__(learning_algorithm, alpha)
        self._train_samples = train_samples
        self._given_features = given_features
        self._new_feature = new_feature
        self._costs_list = costs_list

    def __get_certainty(self):
        """
        return the level of the certainty according to the theory we explain in the PDF.
        :return: the level of the certainty.
        """
        new_features = np.append(self._given_features, [self._new_feature])
        data = pd.DataFrame(self._train_samples.samples)
        self._learning_algorithm.fit(data[new_features], self._train_samples.classes)
        probabilities = self._learning_algorithm.predict_proba(data[new_features])
        certainty = stats.entropy(probabilities[0], probabilities[1])
        return 1 - certainty

    def _execute_function(self):
        price = self._costs_list[self._new_feature]
        certainty = self.__get_certainty()
        return certainty / price
