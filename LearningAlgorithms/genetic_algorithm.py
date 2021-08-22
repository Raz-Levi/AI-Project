""""
This module contains a search algorithm based on a genetic algorithm
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from LearningAlgorithms.abstract_algorithm import SequenceAlgorithm
import sklearn
from General.score import ScoreFunction

from itertools import chain, combinations

""""""""""""""""""""""""""""""""""""""""""" Classes """""""""""""""""""""""""""""""""""""""""""


class GeneticAlgorithm(SequenceAlgorithm):
    """
    A search algorithm based on a genetic algorithm.
    """

    def __init__(self, learning_algorithm: sklearn.base.ClassifierMixin,
                 score_function: ScoreFunction, alpha_for_score_function: [float] = 1):
        super().__init__(learning_algorithm)
        # TODO- check params
        self._score_function = score_function(learning_algorithm, alpha_for_score_function)
        self._features = [i in range(self.)]

    def _buy_features(self, given_features: list[int], maximal_cost: float) -> list[int]:
        pass

    #def _get_population(self, ):
