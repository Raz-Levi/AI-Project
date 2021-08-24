"""
This module contains graph search algorithms.
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from General.utils import *
import networkx as nx
from more_itertools import powerset
import inspect

from LearningAlgorithms.abstract_algorithm import SequenceAlgorithm
from General.score import ScoreFunction

""""""""""""""""""""""""""""""""""" Definitions and Consts """""""""""""""""""""""""""""""""""

node = frozenset[int]
edge = Tuple[node, node]

""""""""""""""""""""""""""""""""""""""""""" Classes """""""""""""""""""""""""""""""""""""""""""


class GraphSearchAlgorithm(SequenceAlgorithm):
    """
    An algorithm that builds features graph and performs search algorithm on this graph.
    """
    # Public Methods
    def __init__(self, learning_algorithm: sklearn.base.ClassifierMixin, search_algorithm: nx.algorithms,
                 score_function: ScoreFunction, algorithm_method: Optional[str] = "dijkstra",
                 alpha_for_score_function: Optional[float] = 1):
        """
        Init function for GraphSearchAlgorithm algorithm.
        :param learning_algorithm: sklearn's classifier. the function saves it and uses it later.
        :param search_algorithm: nx.algorithm for performing search on graph.
        :param score_function: ScoreFunction object for calculating the weights on the edges.
        :param algorithm_method: (Optional) the algorithm to use to compute the path. supported options: ‘dijkstra’, ‘bellman-ford’.
        :param (Optional) alpha_for_score_function: alpha parameter for the ScoreFunction.
        """
        super().__init__(learning_algorithm)
        self._search_algorithm = search_algorithm
        self._algorithm_method = algorithm_method
        self._score_function = score_function(learning_algorithm=learning_algorithm, alpha=alpha_for_score_function)
        self._graph = None

    # Private Methods
    def _buy_features(self, given_features: list[int], maximal_cost: float) -> list[int]:
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

    def _build_graph(self, total_features: int, given_features: list[int]):
        """
        Builds graph for performing search on it as we described in PDF.
        :param total_features: number of the entire features in the train set.
        :param given_features: list of the indices of the chosen features.
        """
        nodes = [frozenset(np.append(sub_set, given_features).astype(np.int)) for sub_set in powerset(get_complementary_set(range(total_features), given_features))]
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(nodes)
        self._graph.add_weighted_edges_from(self._get_edges(nodes))

    def _get_edges(self, nodes: list[node]) -> list[tuple[node, node, float]]:
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

    def _get_shortest_path(self, given_features: list[int]) -> list[node]:
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

    def _fulfill_features(self, given_features: list[int], path: list[node], maximal_cost: float) -> list[int]:
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

    def _features_costs_heuristic(self, node1: node, node2: node) -> float:
        """
        Gets the costs of all the features that are not in node1 and node2. this function is used as heuristic for the
        searches algorithm.
        :param node1: the source of the given edge.
        :param node2: the target of the given edge.
        :return: costs of all the features that are not in node1 and node2.
        """
        return sum(self._features_costs[feature] for feature in get_complementary_set(node2, node1))