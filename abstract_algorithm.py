"""
This module defines an abstract algorithm class.
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
import abc
from utils import *

""""""""""""""""""""""""""""""""""""""""""" Class """""""""""""""""""""""""""""""""""""""""""


class LearningAlgorithm(abc.ABC):
    @abc.abstractmethod
    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        ...

    @abc.abstractmethod
    def predict(self, sample: TestSamples, given_feature: list[int], maximal_cost: float) -> int:
        ...
