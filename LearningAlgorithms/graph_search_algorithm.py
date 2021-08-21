"""
This module contains graph search algorithms.
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from General.utils import *
import networkx as nx
from more_itertools import powerset

from LearningAlgorithms.abstract_algorithm import SequenceAlgorithm
# TODO- check names
from General.score_function.py import ScoreFunction

""""""""""""""""""""""""""""""""""""""""""" Classes """""""""""""""""""""""""""""""""""""""""""


class GraphSearchAlgorithm(SequenceAlgorithm):
    """
    An algorithm that builds features graph and performs search algorithm on this graph.
    """
    # Public Methods
    def __init__(self, learning_algorithm: sklearn.base.ClassifierMixin, search_algorithm: nx.algorithms,
                 score_function: ScoreFunction, alpha_for_score_function: Optional[float] = None):
        super().__init__(learning_algorithm)
        self._search_algorithm = search_algorithm
        # TODO- check params
        self._score_function = score_function(learning_algorithm, alpha_for_score_function)
        self._graph = None

    def _buy_features(self, given_features: list[int], maximal_cost: float) -> list[int]:
        """
        A method for choosing the supplementary features. the method builds a features graph and performs search algorithm
        on this graph. the returned given features are the features in the shortest path that their costs are not above
        the maximal costs.
        a networkx.NetworkXNoPath exception is thrown if no path exists between source and target (this is not supposed
        to happened).
        :param given_features: list of the indices of the chosen features.
        :param maximal_cost: the maximum available cost for buying features.
        :return: the updated given features including all the chosen features.
        """
        self._build_graph(total_features=self._train_samples.samples.shape[1],
                          given_features=given_features,
                          features_costs=self._features_costs)
        path = self._search_algorithm(G=self._graph,
                                      source=frozenset(given_features),
                                      target=frozenset(range(self._train_samples.samples.shape[1])),
                                      heuristic=self._features_costs_heuristic,
                                      weight="weight")
        # TODO- continue this
        # for node in range(1, len(path)):
        #     added_feature = list(get_complementary_list(path[node-1], path[node]))[0]
        #     maximal_cost -= self._features_costs[added_feature - 1]  # TODO check -1
        #     if maximal_cost >= 0:
        #         given_features.append(added_feature)
        #     else:
        #         break
        # return given_features

    # Private Methods
    def _build_graph(self, total_features: int, given_features: list[int], features_costs: list[float]):
        nodes = [frozenset(np.append(sub_set, given_features)) for sub_set in powerset(get_complementary_set(range(total_features), given_features))]
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(nodes)
        self._graph.add_weighted_edges_from(self._get_edges(nodes, features_costs))

    def _get_edges(self, nodes: list[frozenset], features_costs: list[float]) -> list[tuple[frozenset, frozenset, Union[float, list[float]]]]:
        edges = []
        for source in range(len(nodes)):
            for target in range(source, len(nodes)):
                missing_element = list(get_complementary_set(set(nodes[target]), set(nodes[source])))
                if len(nodes[target]) - len(nodes[source]) == 1 and len(missing_element) == 1:
                    # TODO- check params
                    weight = self._score_function(train_samples=self._train_samples,
                                                  features=nodes[source],
                                                  feature=missing_element,
                                                  costs_list=features_costs)
                    edges.append((nodes[source], nodes[target], weight))
        return edges

    def _features_costs_heuristic(self) -> float:
        # TODO- finish this
        # features_costs[missing_element[0] - 1]))
        pass
