"""
This module contains local search algorithm.
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from General.utils import *
from simpleai.search import SearchProblem
from more_itertools import powerset

from LearningAlgorithms.abstract_algorithm import SequenceAlgorithm
from General.score import ScoreFunction

""""""""""""""""""""""""""""""""""" Definitions and Consts """""""""""""""""""""""""""""""""""

Node = frozenset[int]
Edge = Tuple[Node, Node]
State = GivenFeatures


class FeaturesProblem(SearchProblem):
    """
    SimpleAI's search problem for local search algorithm.
    """
    def __init__(self, initial_state: State, train_samples: TrainSamples, score_function: ScoreFunction, total_features: int,
                 maximal_cost: float, features_costs: list[float]):
        """
        Init function for FeaturesProblem.
        :param initial_state: FeaturesProblem's initial state. the local search algorithm will start the searching from this state.
        :param train_samples: training dataset contains training data of shape (n_samples, n_features), i.e samples are
            in the rows, and target values of shape (n_samples,).
        :param score_function: ScoreFunction object for calculating the weights on the edges.
        :param total_features: total features number of the samples in the batch.
        :param maximal_cost: the maximum available cost for buying features.
        :param features_costs: list in length number of features that contains the costs of each feature according to
            indices. in first index you will find the cost of the first feature, etc.
        """
        super().__init__(initial_state)
        self._train_samples = train_samples
        self._score_function = score_function
        self._total_features = total_features
        self._given_features = initial_state
        self._maximal_cost = maximal_cost
        self._features_costs = features_costs
        self._initial_state = initial_state.copy()
        self._scores = {}

    # Public Methods
    def actions(self, state: State) -> List[State]:
        """
        Receives a state, and returns the list of actions that can be performed from that particular state.
        :param state: FeaturesProblem's state.
        :return: list of actions that can be performed from that particular state.
        """
        return [list(np.append(state, feature)) for feature in get_complementary_set(range(self._total_features), state)]

    def result(self, last_state: State, new_state: State) -> State:
        """
        Returns the resulting state of applying that particular action from that particular state.
        :param last_state: FeaturesProblem's state before the action was performed on it.
        :param new_state: FeaturesProblem's state after the action was performed on it.
        :return: resulting state of applying that particular action from that particular state.
        """
        return new_state

    def value(self, state: State) -> float:
        """
        Receives a state, and returns a valuation (“score”) of that value. Better states have higher scores.
        :param state: FeaturesProblem's state.
        :return: the value of the state.
        """
        return self._calculate_score(state) if self._is_valid_state(state) else -np.inf

    def generate_random_state(self) -> State:
        """
        Return a randomly generated state.
        :return: randomly generated state
        """
        states = [list(np.append(sub_set, self._given_features).astype(np.int)) for sub_set in powerset(get_complementary_set(range(self._total_features), self._given_features))]
        return random.choice(states)

    # Private Methods
    def _is_valid_state(self, state: State) -> bool:
        """
        Returns if the state is valid. a valid state is a state that the features in it are below the maximal cost.
        :param state: FeaturesProblem's state.
        :return: if the state is valid or not.
        """
        return sum(self._features_costs[feature] for feature in state) <= self._maximal_cost

    def _calculate_score(self, state: State) -> float:
        """
        Calculates the value of the given state.
        :param state: FeaturesProblem's state.
        :return: the value of the state.
        """
        total_score, states = 0, self._initial_state.copy()
        for new_feature in get_complementary_set(state, self._initial_state):
            if f'{states}+{new_feature}' not in self._scores:
                self._scores[f'{states}+{new_feature}'] = self._score_function(train_samples=self._train_samples,
                                                                               given_features=states,
                                                                               new_feature=new_feature,
                                                                               costs_list=self._features_costs)
            total_score += self._scores[f'{states}+{new_feature}']
            states.append(new_feature)
        return total_score


""""""""""""""""""""""""""""""""""""""""""" Classes """""""""""""""""""""""""""""""""""""""""""


class LocalSearchAlgorithm(SequenceAlgorithm):
    """
    An algorithm that performs local search algorithm on score function.
    """

    # Public Methods
    def __init__(self, classifier: sklearn.base.ClassifierMixin, local_search_algorithm: Callable, score_function: ScoreFunction, **kw):
        """
        Init function for LocalSearchAlgorithm algorithm.
        :param classifier: sklearn's classifier. the function saves it and uses it later.
        :param local_search_algorithm: simpleai's local search algorithm.
        :param score_function: ScoreFunction object for calculating the score of the states.
        """
        super().__init__(classifier)
        self._local_search_algorithm = local_search_algorithm
        self._score_function = score_function
        self._kw = kw

    # Private Methods
    def _buy_features(self, given_features: GivenFeatures, maximal_cost: float) -> GivenFeatures:
        """
        A method for choosing the supplementary features. the method performs local search algorithm.
        the returned given features are the features in the maximal state that their costs are not above the maximal costs.
        :param given_features: list of the indices of the chosen features.
        :param maximal_cost: the maximum available cost for buying features.
        :return: the updated given features including all the chosen features.
        """
        best_state = self._get_best_state(given_features, maximal_cost)
        return sorted(best_state)

    def _get_best_state(self, given_features: GivenFeatures, maximal_cost: float) -> State:
        """
        Performs local search algorithm on the score function.
        :param given_features: list of the indices of the chosen features.
        :param maximal_cost: the maximum available cost for buying features.
        :return: the best state that was found by the algorithm.
        """
        initial_state = FeaturesProblem(initial_state=given_features,
                                        train_samples=self._train_samples,
                                        score_function=self._score_function,
                                        total_features=self._train_samples.get_features_num(),
                                        maximal_cost=maximal_cost,
                                        features_costs=self._features_costs)
        return self._local_search_algorithm(initial_state, **self._kw).state
