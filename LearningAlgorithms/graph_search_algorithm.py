"""
This module contains graph search algorithms.
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from General.utils import *
import networkx as nx
from more_itertools import powerset

from LearningAlgorithms.abstract_algorithm import LearningAlgorithm

""""""""""""""""""""""""""""""""""""""""""" Classes """""""""""""""""""""""""""""""""""""""""""


class GraphSearchAlgorithm(LearningAlgorithm):
    """
    An algorithm that builds features graph and perform search algorithm on this graph.
    """
    # Public Methods
    def __init__(self, search_algorithm: nx.algorithms):
        super().__init__()
        self._search_algorithm = search_algorithm
        self._graph = None
        self._train_samples = None
        self._features_costs = None

    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        """
        A method for training the classifier.
        :param train_samples: training dataset contains training data of shape (n_samples, n_features), i.e samples are
        in the rows, and target values of shape (n_samples,).
        :param features_costs: list in length number of features that contains the costs of each feature according to
        indices. in first index you will find the cost of the first feature, etc.
        """
        self._train_samples = train_samples
        self._features_costs = features_costs

    def predict(self, samples: TestSamples, given_features: list[int], maximal_cost: float) -> Classes:
        """
        A method for predicting the class labels for the provided data.
        :param samples: test samples of shape (n_samples, n_features), i.e samples are in the rows.
        :param given_features: list of the indices of the chosen features.
        :param maximal_cost: the maximum available cost for buying features.
        :return: Classes of shape (n_samples,) contains the class labels for each data sample.
        """
        self._build_graph(total_features=self._train_samples.samples.shape[1],
                          given_features=given_features,
                          features_costs=self._features_costs)

    # Private Methods
    def _build_graph(self, total_features: int, given_features: list[int], features_costs: list[float]):
        nodes = [frozenset(np.append(sub_set, given_features)) for sub_set in powerset(get_complementary_numbers(total_features, given_features))]
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(nodes)
        self._graph.add_weighted_edges_from(self._get_edges(nodes, features_costs))

    @staticmethod
    def _get_edges(nodes: list[frozenset], features_costs: list[float]) -> list[tuple[frozenset, frozenset, Union[float, list[float]]]]:
        edges = []
        for source in range(len(nodes)):
            for target in range(source, len(nodes)):
                missing_element = list(set(nodes[target]) - set(nodes[source]))  # TODO- make it a function of get_complementary_numbers
                if len(nodes[target]) - len(nodes[source]) == 1 and len(missing_element) == 1:
                    edges.append((nodes[source], nodes[target], features_costs[missing_element[0] - 1]))
        return edges
