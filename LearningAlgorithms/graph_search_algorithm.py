"""
This module contains search algorithms.
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from General.utils import *
from simpleai.search import SearchProblem
from more_itertools import powerset
import inspect

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


class GraphSearchAlgorithm(SequenceAlgorithm):
    """
    An algorithm that builds features graph and performs search algorithm on this graph.
    """
    # Public Methods
    def __init__(self, classifier: sklearn.base.ClassifierMixin, search_algorithm: nx.algorithms,
                 score_function: ScoreFunction, algorithm_method: Optional[str] = "dijkstra"):
        """
        Init function for GraphSearchAlgorithm algorithm.
        :param classifier: sklearn's classifier. the function saves it and uses it later.
        :param search_algorithm: nx.algorithm for performing search on graph.
        :param score_function: ScoreFunction object for calculating the weights on the edges.
        :param algorithm_method: (Optional) the algorithm to use to compute the path. supported options: ‘dijkstra’, ‘bellman-ford’.
        """
        super().__init__(classifier)
        self._search_algorithm = search_algorithm
        self._algorithm_method = algorithm_method
        self._score_function = score_function
        self._graph = None

    # Private Methods
    def _buy_features(self, given_features: GivenFeatures, maximal_cost: float) -> GivenFeatures:
        """
        A method for choosing the supplementary features. the method builds a features graph and performs search algorithm
        on this graph. the returned given features are the features in the shortest path that their costs are not above
        the maximal costs.
        :param given_features: list of the indices of the chosen features.
        :param maximal_cost: the maximum available cost for buying features.
        :return: the updated given features including all the chosen features.
        """
        self._build_graph(total_features=self._train_samples.get_features_num(), given_features=given_features)
        path = self._get_shortest_path(given_features)
        return self._fulfill_features(given_features, path, maximal_cost)

    def _build_graph(self, total_features: int, given_features: GivenFeatures):
        """
        Builds graph for performing search on it as we described in PDF.
        :param total_features: number of the entire features in the train set.
        :param given_features: list of the indices of the chosen features.
        """
        nodes = [frozenset(np.append(sub_set, given_features).astype(np.int)) for sub_set in powerset(get_complementary_set(range(total_features), given_features))]
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(nodes)
        self._graph.add_weighted_edges_from(self._get_edges(nodes))

    def _get_edges(self, nodes: list[Node]) -> list[tuple[Node, Node, float]]:
        """
        Gets the edges of the graph and their weights.
        :param nodes: list of the nodes in the graph.
        :return: tuple of the edges and their weights in form [source, target, weight].
        """
        edges = []
        for source in range(len(nodes)):
            for target in range(source, len(nodes)):
                missing_feature = get_complementary_set(nodes[target], nodes[source])
                if len(missing_feature) == 1 and len(nodes[target]) - len(nodes[source]) == 1:
                    weight = self._score_function(train_samples=self._train_samples,
                                                  given_features=list(nodes[source]),
                                                  new_feature=missing_feature.pop(),
                                                  costs_list=self._features_costs)
                    edges.append((nodes[source], nodes[target], weight))
        return edges

    def _get_shortest_path(self, given_features: GivenFeatures) -> list[Node]:
        """
        Executes the searching algorithm.
        :param given_features: list of the indices of the chosen features.
        :return: list of nodes in a shortest path.
        """
        if 'method' in inspect.signature(self._search_algorithm).parameters.keys():
            return self._search_algorithm(G=self._graph,
                                          source=frozenset(given_features),
                                          target=frozenset(range(self._train_samples.get_features_num())),
                                          method=self._algorithm_method,
                                          weight="weight")
        return self._search_algorithm(G=self._graph,
                                      source=frozenset(given_features),
                                      target=frozenset(range(self._train_samples.get_features_num())),
                                      heuristic=self._features_costs_heuristic,
                                      weight="weight")

    def _fulfill_features(self, given_features: GivenFeatures, path: list[Node], maximal_cost: float) -> GivenFeatures:
        """
        Returns the features in the shortest path that their costs are not above the maximal costs.
        :param path: list of nodes in a shortest path.
        :return: the updated given features including all the chosen features.
        """
        for vertex in range(1, len(path)):
            added_feature = get_complementary_set(path[vertex], path[vertex-1]).pop()
            maximal_cost -= self._features_costs[added_feature]
            if maximal_cost >= 0:
                given_features.append(added_feature)
            else:
                break
        return given_features

    def _features_costs_heuristic(self, node1: Node, node2: Node) -> float:
        """
        Gets the costs of all the features that are not in node1 and node2. this function is used as heuristic for the
        searches algorithm.
        :param node1: the source of the given edge.
        :param node2: the target of the given edge.
        :return: costs of all the features that are not in node1 and node2.
        """
        return sum(self._features_costs[feature] for feature in get_complementary_set(node2, node1))


class LocalSearchAlgorithm(SequenceAlgorithm):
    """
    An algorithm that performs local search algorithm on score function.
    """

    # Public Methods
    def __init__(self, classifier: sklearn.base.ClassifierMixin, local_search_algorithm: Callable, score_function: ScoreFunction):
        """
        Init function for LocalSearchAlgorithm algorithm.
        :param classifier: sklearn's classifier. the function saves it and uses it later.
        :param local_search_algorithm: simpleai's local search algorithm.
        :param score_function: ScoreFunction object for calculating the score of the states.
        """
        super().__init__(classifier)
        self._local_search_algorithm = local_search_algorithm
        self._score_function = score_function

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
        return self._local_search_algorithm(problem=initial_state).state
