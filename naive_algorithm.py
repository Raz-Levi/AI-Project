"""
This module contains naive algorithms
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from utils import *
from abstract_algorithm import LearningAlgorithm

""""""""""""""""""""""""""""""""""""""""""" Classes """""""""""""""""""""""""""""""""""""""""""


class EmptyAlgorithm(LearningAlgorithm):
    def __init__(self, learning_algorithm: sklearn.base.ClassifierMixin):
        super().__init__()
        self._learning_algorithm = learning_algorithm

    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        self._learning_algorithm.fit(train_samples.samples, train_samples.classes)

    def predict(self, sample: TestSamples, given_features: list[int], maximal_cost: float) -> int:
        return self._learning_algorithm.predict(complete_features(sample, given_features))


class RandomAlgorithm(LearningAlgorithm):
    def __init__(self, learning_algorithm: sklearn.base.ClassifierMixin):
        super().__init__()
        self._learning_algorithm = learning_algorithm

    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        self._learning_algorithm.fit(train_samples.samples, train_samples.classes)

    def predict(self, sample: TestSamples, given_features: list[int], maximal_cost: float) -> int:
        return self._learning_algorithm.predict(complete_features(sample, given_features))
