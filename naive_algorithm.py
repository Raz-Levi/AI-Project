"""
This module contains naive algorithms
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
import random

from utils import *
from abstract_algorithm import LearningAlgorithm

""""""""""""""""""""""""""""""""""""""""""" Classes """""""""""""""""""""""""""""""""""""""""""


class EmptyAlgorithm(LearningAlgorithm):
    def __init__(self, learning_algorithm: sklearn.base.ClassifierMixin):
        super().__init__()
        self._train_samples = None
        self._learning_algorithm = learning_algorithm

    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        self._train_samples = train_samples

    def predict(self, samples: TestSamples, given_features: list[int], maximal_cost: float) -> Classes:
        self._learning_algorithm.fit(self._train_samples.samples[:, given_features], self._train_samples.classes)
        return self._learning_algorithm.predict(samples[:, given_features])


class SequenceAlgorithm(LearningAlgorithm):
    # Public Methods
    def __init__(self, learning_algorithm: sklearn.base.ClassifierMixin):
        super().__init__()
        self._train_samples = None
        self._features_costs = []
        self._learning_algorithm = learning_algorithm

    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        self._train_samples = train_samples
        self._features_costs = features_costs

    def predict(self, samples: TestSamples, given_features: list[int], maximal_cost: float) -> Classes:
        given_features = self._buy_features(given_features, maximal_cost)
        self._learning_algorithm.fit(self._train_samples.samples[:, given_features], self._train_samples.classes)
        return self._learning_algorithm.predict(samples[:, given_features])

    # Private Methods
    @abc.abstractmethod
    def _buy_features(self, given_features: list[int], maximal_cost: float) -> list[int]:
        ...


class RandomAlgorithm(SequenceAlgorithm):
    def __init__(self, learning_algorithm: sklearn.base.ClassifierMixin):
        super().__init__(learning_algorithm)

    # Private Methods
    def _buy_features(self, given_features: list[int], maximal_cost: float) -> list[int]:
        available_features = set(range(self._train_samples.samples.shape[1])) - set(given_features)
        while len(available_features) and maximal_cost:
            chosen_feature = random.choice(list(available_features))
            available_features -= {chosen_feature}
            if self._features_costs[chosen_feature] <= maximal_cost:
                maximal_cost -= self._features_costs[chosen_feature]
                given_features.append(chosen_feature)
        return given_features


class OptimalAlgorithm(SequenceAlgorithm):
    def __init__(self, learning_algorithm: sklearn.base.ClassifierMixin):
        super().__init__(learning_algorithm)

    # Private Methods
    def _buy_features(self, given_features: list[int], maximal_cost: float) -> list[int]:
        available_features = list(set(range(self._train_samples.samples.shape[1])) - set(given_features))
        available_features.sort(reverse=True, key=lambda a: self._features_costs[a])
        while len(available_features) and maximal_cost:
            chosen_feature = available_features.pop()
            if self._features_costs[chosen_feature] > maximal_cost:
                break
            maximal_cost -= self._features_costs[chosen_feature]
            given_features.append(chosen_feature)
        return given_features
