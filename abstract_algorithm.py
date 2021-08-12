"""
This module defines an abstract algorithm class.
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
import abc
from utils import *

""""""""""""""""""""""""""""""""""""""""""" Class """""""""""""""""""""""""""""""""""""""""""


class LearningAlgorithm(abc.ABC):
    def __init__(self):
        self._total_features_num = None

    ### Public Methods ###
    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        self._total_features_num = train_samples.samples.shape[0]
        self._fit(train_samples, features_costs)

    @abc.abstractmethod
    def predict(self, sample: TestSamples, given_features: list[int], maximal_cost: float) -> int:
        ...

    ### Private Methods ###
    @abc.abstractmethod
    def _fit(self, train_samples: TrainSamples, features_costs: list[float]):
        ...
