""""
This module contains a search algorithm based on a genetic algorithm
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from LearningAlgorithms.abstract_algorithm import SequenceAlgorithm
import sklearn
from General.score import ScoreFunction

from itertools import chain, combinations
from General.utils import get_complementary_set

""""""""""""""""""""""""""""""""""""""""""" Classes """""""""""""""""""""""""""""""""""""""""""


class GeneticAlgorithm(SequenceAlgorithm):
    """
    A search algorithm based on a genetic algorithm.
    """

    def __init__(self, number_of_features: int, learning_algorithm: sklearn.base.ClassifierMixin,
                 score_function: ScoreFunction, alpha_for_score_function: [float] = 1):
        super().__init__(learning_algorithm)
        # TODO- check params
        self._score_function = score_function(learning_algorithm, alpha_for_score_function)
        self._all_features = [i for i in range(number_of_features)]

    def _buy_features(self, given_features: list[int], maximal_cost: float) -> list[int]:
        pass

    def _get_population(self, given_features: list[int], maximal_cost: float):
        """
        this function wil create a proper population. a member in the population is a subset of features, which total
        cost is less equal than maximal cost
        :param given_features: the features that are given us for "free"
        :param maximal_cost: the maximal cost for the learning process
        :return: population
        """
        tested_features = get_complementary_set(self._all_features, given_features)
        power_series = chain.from_iterable(combinations(tested_features, item) for item in range(len(tested_features) + 1))
        population = []
        for subset in power_series:
            if self._calc_subset_cost(list(subset)) <= maximal_cost:
                population.append(subset)
        return population

    def _calc_subset_cost(self, subset: list[int]) -> float:
        total_cost = 0
        for item in subset:
            total_cost += self._features_costs[item]
        return total_cost

