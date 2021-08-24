"""
Automation Tests For The Project
"""
from General.score import ScoreFunctionA
from LearningAlgorithms.genetic_algorithm import GeneticAlgorithm

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
import unittest
from General.utils import *
from sklearn.neighbors import KNeighborsClassifier
from networkx.algorithms.shortest_paths.astar import astar_path

from LearningAlgorithms.abstract_algorithm import LearningAlgorithm
from LearningAlgorithms.naive_algorithm import EmptyAlgorithm, RandomAlgorithm, OptimalAlgorithm
from LearningAlgorithms.mid_algorithm import MaxVarianceAlgorithm
#from LearningAlgorithms.graph_search_algorithm import GraphSearchAlgorithm

""""""""""""""""""""""""""""""""""""""""" Tests  """""""""""""""""""""""""""""""""""""""""


class TestUtils(unittest.TestCase):
    # tests functions
    def test_get_samples_from_csv(self):
        consts = self._get_consts()
        self._test_get_samples_from_csv(path=consts["csv_path"], expected_matrix=consts["full_expected_matrix"])

    def test_categorical_to_numeric(self):
        consts, categories = self._get_consts(), {}
        self._test_get_samples_from_csv(path=consts["csv_with_strings_path"],
                                        expected_matrix=consts["csv_strings_expected_matrix"],
                                        preprocess=categorical_to_numeric,
                                        categories=categories)
        categories = {}
        self._test_get_samples_from_csv(path=consts["csv_few_samples"],
                                        expected_matrix=consts["csv_samples_expected_matrix"],
                                        preprocess=categorical_to_numeric,
                                        categories=categories)

    def test_get_dataset(self):
        consts = self._get_consts()
        for ratio in consts["train_ratio"]:
            self._test_get_dataset(path=consts["csv_path"], expected_matrix=consts["full_expected_matrix"], train_ratio=ratio, random_seed=consts["random_seed"])
            self._test_get_dataset(path=consts["csv_with_strings_path"], expected_matrix=consts["csv_strings_expected_matrix"], train_ratio=ratio, random_seed=consts["random_seed"])

    def test_declarations(self):
        consts = self._get_consts()
        sample = consts["sample"]
        classes = consts["classes"]
        train_samples = TrainSamples(sample, classes)
        self.assertTrue(np.array_equal(train_samples.samples, sample))
        self.assertTrue(np.array_equal(train_samples.classes, classes))

    def test_complete_features(self):
        consts = self._get_consts()
        self.assertTrue(np.array_equal(complete_features(samples=consts["sample"],
                                                         given_features=consts["given_features"],
                                                         total_features_num=consts["total_features_num"]),
                                       consts["completed_features_inf"]))
        self.assertTrue(np.array_equal(complete_features(samples=consts["sample"],
                                                         given_features=consts["given_features"],
                                                         total_features_num=consts["total_features_num"],
                                                         default_value=consts["default_value"]),
                                       consts["completed_features_zero"]))
        self.assertTrue(np.array_equal(complete_features(samples=consts["sample"],
                                                         given_features=consts["given_features_not_sorted"],
                                                         total_features_num=consts["total_features_num"],
                                                         default_value=consts["default_value"]),
                                       consts["completed_features_not_sorted"]))
        self.assertTrue(np.array_equal(complete_features(samples=consts["full_sample"],
                                                         given_features=consts["given_features_full"],
                                                         total_features_num=consts["total_features_num"]),
                                       consts["completed_features_full"]))

    def test_get_correlation_to_feature(self):
        consts = self._get_consts()
        data = normalize_data(consts["corr_matrix"])
        corr = get_correlation_to_feature(data, 0, 1)
        self.assertEqual(1, corr)
        corr = get_correlation_to_feature(data, 0, 2)
        self.assertEqual(0, corr)

    def test_get_correlation_to_other_features(self):
        consts = self._get_consts()
        data = normalize_data(consts["corr_matrix"])
        corr = get_correlation_to_other_features(data, [1, 2], 0)
        self.assertEqual(0.5, corr)

    def test_get_price_score(self):
        consts = self._get_consts()
        costs_list = consts["costs_list"]
        self.assertEqual(1, get_price_score(0, costs_list))
        self.assertEqual(2, get_price_score(1, costs_list))

    def test_get_get_certainty(self):
        consts = self._get_consts()
        data = normalize_data(consts["corr_matrix"])
        learner = KNeighborsClassifier(1)
        res = get_certainty(data, 3, [0, 1], 2, learner)
        self.assertEqual(1, res)
        res = get_certainty(data, 3, [2, 1], 0, learner)

    def test_score_function_a(self):
        consts = self._get_consts()
        data = normalize_data(consts["corr_matrix"])
        costs_list = consts["costs_list"]
        res = score_function_a(data, [2], 1, 0, costs_list, alpha=1)
        self.assertEqual(0, res)
        res = score_function_a(data, [1, 2], 3, 0, costs_list, alpha=2)
        self.assertEqual(consts["score"], res)

    def test_score_function_b(self):
        consts = self._get_consts()
        data = normalize_data(consts["corr_matrix"])
        costs_list = consts["costs_list"]
        learner = KNeighborsClassifier(1)
        res = score_function_b(data, [1, 2], 0, 3, costs_list, learning_algo=learner)
        self.assertEqual(1, res)
        res = score_function_b(data, [1, 0], 1, 3, costs_list, learning_algo=learner)
        self.assertEqual(0.5, res)

    # private functions
    def _test_get_samples_from_csv(self, path: str, expected_matrix: np.array, preprocess: Callable = None, **kw):
        for col in range(expected_matrix.shape[1]):
            samples, classes = get_samples_from_csv(path=path, class_index=col, preprocess=preprocess, **kw)
            self.assertTrue(self._compare_samples(samples=samples,
                                                  classes=classes,
                                                  expected_matrix=expected_matrix,
                                                  class_index=col,
                                                  **kw))

    def _test_get_dataset(self, path: str, expected_matrix: np.array, train_ratio, random_seed: int):
        for col in range(expected_matrix.shape[1]):
            train_samples, test_samples = get_dataset(path=path, class_index=col, train_ratio=train_ratio, random_seed=random_seed, shuffle=False)
            tested_rows = list(range(expected_matrix.shape[0]))[-train_ratio:]
            complementary_list = list(get_complementary_set(range(train_samples.samples.shape[0]), tested_rows))
            self.assertTrue(self._compare_samples(train_samples.samples, train_samples.classes, expected_matrix[complementary_list, :], col))
            self.assertTrue(self._compare_samples(test_samples.samples, test_samples.classes, expected_matrix[tested_rows, :], col))

    @staticmethod
    def _get_consts() -> dict:
        return {
            "csv_path": "test_csv_functions.csv",
            "csv_with_strings_path": "test_csv_with_strings.csv",
            "csv_few_samples": "test_csv_few_samples.csv",
            "corr_matrix": [[1, 2, 3, 0], [-2, -4, -6, 0], [3, 6, -5, 1]],
            "costs_list": [1, 2, 3, 4],
            "sample": np.array([[2, 2, 2]]),
            "classes": np.array([1]),
            "full_sample": np.array([[0, 1, 2, 3, 4, 5, 6]]),
            "given_features": np.array([2, 4, 6]),
            "given_features_not_sorted": np.array([6, 0, 2]),
            "given_features_full": np.array([5, 1, 2, 0, 6, 4, 3]),
            "total_features_num": 7,
            "default_value": 0,
            "random_seed": 0,
            "train_ratio": [1, 2, 3, 4],
            "completed_features_inf": np.array([[np.inf, np.inf, 2, np.inf, 2, np.inf, 2]]),
            "completed_features_zero": np.array([[0, 0, 2, 0, 2, 0, 2]]),
            "completed_features_not_sorted": np.array([[2, 0, 2, 0, 0, 0, 2]]),
            "completed_features_full": np.array([[3, 1, 2, 6, 5, 0, 4]]),
            "score": 0.07012591041294361,
            "full_expected_matrix": np.array(
                [[1, 0.11, 0.05, 78, 32, 12, 4231], [0, 3.6, 5.4, 4.32, 432.2, 21.4, 43.21],
                 [1, 2, 0, 43, 21, 245, 4.231], [1, 22, 32, 6, 3.45, 62.4, 2.2], [62, 32, 12, 214, 215, 53.215, 21]]),
            "csv_strings_expected_matrix": np.array(
                [[0.2, 0., 0.05, 0., 0., 0., 0.9, 0, 0, 3], [1., 0., 5.4, 1., 0., 1., 1.2, 1, 1, 10],
                 [1., 1., 0., 0., 1., 1., 5.6, 0, 1, 20], [0.1, 1, 0.4, 0., 1., 0., 0., 1, 0, 10],
                 [0.3, 0, 0.9, 1, 1, 0, 1.8, 1, 1, 3], [0.4, 0, 1.2, 1, 0, 1, 0.3, 0, 0, 20]]),
            "csv_samples_expected_matrix": np.array(
                [[0, 0, 0, 13, 0, 0, 460, 3, 4, 0], [1, 0, 1, 25, 1, 1, 235, 3, 2, 0],
                 [2, 1, 0, 26, 1, 1, 1142, 2, 2, 1]])
        }

    @staticmethod
    def _compare_samples(samples: np.array, classes: np.array, expected_matrix: np.array, class_index: int, **kw) -> bool:
        complementary_list = list(get_complementary_set(range(expected_matrix.shape[1]), [class_index]))
        expected_samples, expected_classes = expected_matrix[:, complementary_list], expected_matrix[:, [class_index]]
        return type(samples) == np.ndarray and np.array_equal(samples, expected_samples) and type(
            classes) == np.ndarray and np.array_equal(classes, expected_classes.flatten())


class TestLearningAlgorithm(unittest.TestCase):
    # tests functions
    def test_initialization(self):
        consts = self._get_consts()
        simple_algorithm = self._get_instance()
        self.assertTrue(simple_algorithm.predict(sample=consts["test_sample"], given_feature=consts["given_feature"],
                                                 maximal_cost=consts["test_sample"]))

    def test_fit(self):
        consts = self._get_consts()
        simple_algorithm = self._get_instance()
        self.assertTrue(simple_algorithm._get_total_features_num() is None)
        simple_algorithm.fit(consts["train_samples"], consts["features_costs"])
        self.assertTrue(simple_algorithm._get_total_features_num() == consts["total_features_num"])

    # private functions
    @staticmethod
    def _get_consts() -> dict:
        return {
            "test_sample": np.array([1]),
            "train_samples": TrainSamples(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]]),
                                          np.array([1, 1, 1, 1])),
            "given_feature": [1],
            "total_features_num": 3,
            "features_costs": [1],
            "maximal_cost": 1
        }

    @staticmethod
    def _get_instance() -> LearningAlgorithm:
        class SimpleAlgorithm(LearningAlgorithm):
            def __init__(self):
                self._total_features_num = None

            def fit(self, train_samples: TrainSamples, features_costs: list[float]):
                self._total_features_num = train_samples.samples.shape[1]

            def predict(self, sample: TestSamples, given_feature: list[int], maximal_cost: float) -> int:
                return True

            def _get_total_features_num(self) -> int:
                return self._total_features_num

        return SimpleAlgorithm()


class TestNaiveAlgorithm(unittest.TestCase):
    # tests functions
    def test_initializations(self):
        self.assertTrue(self._test_initialization(EmptyAlgorithm))
        self.assertTrue(self._test_initialization(RandomAlgorithm))
        self.assertTrue(self._test_initialization(OptimalAlgorithm))
        self.assertTrue(self._test_initialization(MaxVarianceAlgorithm))

    def test_algorithms(self):
        self.assertTrue(self._test_naive_algorithm(EmptyAlgorithm)[0])

    def test_random_algorithm(self):
        self.assertTrue(self._test_sequence_algorithm(RandomAlgorithm)[0])

    def test_optimal_algorithm(self):
        self.assertTrue(self._test_sequence_algorithm(OptimalAlgorithm)[0])

    def test_mid_algorithm(self):
        self.assertTrue(self._test_mid_algorithm(MaxVarianceAlgorithm)[0])

    # private functions
    @staticmethod
    def _get_consts() -> dict:
        return {
            "learning_algorithm": KNeighborsClassifier(n_neighbors=1),
            "train_samples": TrainSamples(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]]),
                                          np.array([0, 1, 0, 1])),
            "features_costs": [1, 5, 2],
            "maximal_cost": 10,
            "maximal_cost_zero": 0,
            "maximal_cost_partially": 4,
            "given_features_full": [0, 1, 2],
            "given_features_missed": [0, 1],
            "given_features_empty": []
        }

    @staticmethod
    def _test_initialization(tested_algorithm) -> bool:
        consts = TestNaiveAlgorithm._get_consts()
        algorithm = tested_algorithm(learning_algorithm=consts["learning_algorithm"])
        return type(algorithm) == tested_algorithm and hasattr(algorithm.predict, '__call__') and hasattr(algorithm.fit, '__call__')

    @staticmethod
    def _test_naive_algorithm(tested_algorithm) -> Tuple[bool, LearningAlgorithm]:
        consts = TestNaiveAlgorithm._get_consts()
        algorithm = tested_algorithm(learning_algorithm=consts["learning_algorithm"])
        test_result = type(algorithm) == tested_algorithm
        algorithm.fit(train_samples=consts["train_samples"], features_costs=consts["features_costs"])

        predicted_sample = algorithm.predict(samples=consts["train_samples"].samples,
                                             given_features=consts["given_features_full"],
                                             maximal_cost=consts["maximal_cost"])
        test_result = test_result and np.array_equal(predicted_sample, consts["train_samples"].classes)

        predicted_sample = algorithm.predict(samples=np.array([consts["train_samples"].samples[0]]),
                                             given_features=consts["given_features_missed"],
                                             maximal_cost=consts["maximal_cost"])
        test_result = test_result and np.array_equal(predicted_sample.item(), consts["train_samples"].classes[0])

        predicted_sample = algorithm.predict(samples=consts["train_samples"].samples,
                                             given_features=consts["given_features_missed"],
                                             maximal_cost=consts["maximal_cost"])
        test_result = test_result and np.array_equal(predicted_sample, consts["train_samples"].classes)
        return test_result, algorithm

    @staticmethod
    def _test_sequence_algorithm(tested_algorithm) -> Tuple[bool, LearningAlgorithm]:
        consts = TestNaiveAlgorithm._get_consts()
        test_result, algorithm = TestNaiveAlgorithm._test_naive_algorithm(tested_algorithm)
        predicted_sample = algorithm.predict(samples=consts["train_samples"].samples,
                                             given_features=consts["given_features_empty"],
                                             maximal_cost=consts["maximal_cost_partially"])
        test_result = test_result and np.array_equal(predicted_sample, consts["train_samples"].classes)
        return test_result, algorithm

    @staticmethod
    def _test_mid_algorithm(tested_algorithm) -> Tuple[bool, LearningAlgorithm]:
        consts = TestNaiveAlgorithm._get_consts()
        test_result, algorithm = TestNaiveAlgorithm._test_naive_algorithm(tested_algorithm)
        predicted_sample = algorithm.predict(samples=consts["train_samples"].samples,
                                             given_features=consts["given_features_missed"],
                                             maximal_cost=consts["maximal_cost_partially"])
        test_result = test_result and np.array_equal(predicted_sample, consts["train_samples"].classes)
        return test_result, algorithm


class TestGeneticAlgorithm(unittest.TestCase):
    # tests functions
    def test_initialization(self):
        algorithm = GeneticAlgorithm(10, KNeighborsClassifier(n_neighbors=1), ScoreFunctionA())
        return type(algorithm) == GeneticAlgorithm

    def test_buy_features(self):
        algorithm = GeneticAlgorithm(6, KNeighborsClassifier(n_neighbors=1), ScoreFunctionA())
        consts = self._get_consts()
        train_samples, _ = get_dataset(consts["numeric_samples_path"], train_ratio=consts["train_ratio"], class_index=12)
        algorithm.fit(train_samples, consts["features_costs"])
        res = algorithm._buy_features(consts["given_features"][0], consts["maximal_cost"])
        self.assertTrue(algorithm._is_legal_subset(res))

    # private functions
    @staticmethod
    def _get_consts() -> dict:
        return {
            "learning_algorithm": KNeighborsClassifier(n_neighbors=1),
            "numeric_samples_path": "heart_failure_clinical_records_dataset.csv",
            "train_ratio": 1,
            "features_costs": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 22, 23],
            "given_features": [[0], [3], [2, 3]],
            "maximal_cost": 10,
            "total_features": 12,
            }

#
# class TestGraphSearchAlgorithm(unittest.TestCase):
#     # tests functions
#     def test_initialization(self):
#         consts = self._get_consts()
#         algorithm = GraphSearchAlgorithm(consts["learning_algorithm"], consts["search_algorithm"], consts["heuristic"])
#         return type(algorithm) == GraphSearchAlgorithm and hasattr(algorithm.predict, '__call__') and hasattr(algorithm.fit, '__call__')
#
#     def test_build_graph(self):
#         consts = self._get_consts()
#         algorithm = GraphSearchAlgorithm(consts["learning_algorithm"], consts["search_algorithm"], consts["heuristic"])
#         for given_features in consts["given_features"]:
#             features_costs = [i for i in range(len(get_complementary_set(range(consts["total_features"]), given_features)))]
#             algorithm._build_graph(total_features=consts["total_features"], given_features=given_features, features_costs=features_costs)
#             self.assertTrue(algorithm._graph.nodes, consts[f'expected_nodes_{given_features}'])
#             self.assertTrue(algorithm._graph.edges, consts[f'expected_nodes_{given_features}'])
#             self.assertTrue(list(algorithm._graph.nodes)[0] == frozenset(given_features))
#             self.assertTrue(list(algorithm._graph.nodes)[-1] == frozenset(range(consts["total_features"])))
#
#     # TODO- finish this test
#     def test_buy_features(self):
#         print("\n")
#         consts = self._get_consts()
#         train_samples, _ = get_dataset(consts["numeric_samples_path"], train_ratio=consts["train_ratio"])
#         algorithm = GraphSearchAlgorithm(consts["learning_algorithm"], consts["search_algorithm"], consts["heuristic"])
#         algorithm.fit(train_samples, consts["features_costs"])
#         print(algorithm._buy_features(consts["given_features"][0], consts["maximal_cost"]))
#
#     # TODO- finish this test
#     # def test_graph_search_algorithm(self):
#     #     consts = self._get_consts()
#     #     train_samples, test_samples = get_dataset(consts["numeric_samples_path"], train_ratio=consts["train_ratio"])
#     #     algorithm = GraphSearchAlgorithm(consts["learning_algorithm"], consts["search_algorithm"], consts["heuristic"])
#     #     algorithm.fit(train_samples, consts["features_costs"])
#     #     algorithm.predict(test_samples, consts["given_features"][0], consts["maximal_cost"])
#
#     # private functions
#     @staticmethod
#     def _get_consts() -> dict:
#         def simple_heuristic(a, b) -> float:
#             print(a)
#             print(b)
#             print("\n")
#             print("\n")
#             return 0.2
#
#         return {
#             "learning_algorithm": KNeighborsClassifier(n_neighbors=1),
#             "search_algorithm": astar_path,
#             "heuristic": simple_heuristic,
#             "numeric_samples_path": "test_csv_functions.csv",
#             # "strings_samples_path": "test_csv_with_strings.csv",
#             "train_ratio": 1,
#             "features_costs": [1, 2, 3, 4, 5, 6],
#             "given_features": [[0], [3], [2, 3]],
#             "maximal_cost": 10,
#             "total_features": 4,
#             "expected_nodes_[0]": [frozenset({0.0}), frozenset({0, 1}), frozenset({0, 2}), frozenset({0, 3}),
#                                    frozenset({0, 1, 2}), frozenset({0, 1, 3}), frozenset({0, 2, 3}), frozenset({0, 1, 2, 3})],
#             "expected_edges_[0]": [(frozenset({0.0}), frozenset({0, 1})), (frozenset({0.0}), frozenset({0, 2})),
#                                    (frozenset({0.0}), frozenset({0, 3})), (frozenset({0, 1}), frozenset({0, 1, 2})),
#                                    (frozenset({0, 1}), frozenset({0, 1, 3})), (frozenset({0, 2}), frozenset({0, 1, 2})),
#                                    (frozenset({0, 2}), frozenset({0, 2, 3})), (frozenset({0, 3}), frozenset({0, 1, 3})),
#                                    (frozenset({0, 3}), frozenset({0, 2, 3})), (frozenset({0, 1, 2}), frozenset({0, 1, 2, 3})),
#                                    (frozenset({0, 1, 3}), frozenset({0, 1, 2, 3})), (frozenset({0, 2, 3}), frozenset({0, 1, 2, 3}))],
#             "expected_nodes_[3]": [frozenset({3.0}), frozenset({0, 3}), frozenset({1, 3}), frozenset({2, 3}), frozenset({0, 1, 3}),
#                                    frozenset({0, 2, 3}), frozenset({1, 2, 3}), frozenset({0, 1, 2, 3})],
#             "expected_edges_[3]": [(frozenset({3.0}), frozenset({0, 3})), (frozenset({3.0}), frozenset({1, 3})), (frozenset({3.0}),
#                                    frozenset({2, 3})), (frozenset({0, 3}), frozenset({0, 1, 3})), (frozenset({0, 3}), frozenset({0, 2, 3})),
#                                    (frozenset({1, 3}), frozenset({0, 1, 3})), (frozenset({1, 3}), frozenset({1, 2, 3})), (frozenset({2, 3}),
#                                    frozenset({0, 2, 3})), (frozenset({2, 3}), frozenset({1, 2, 3})), (frozenset({0, 1, 3}), frozenset({0, 1, 2, 3})),
#                                    (frozenset({0, 2, 3}), frozenset({0, 1, 2, 3})), (frozenset({1, 2, 3}), frozenset({0, 1, 2, 3}))],
#             "expected_nodes_[2, 3]": [frozenset({2.0, 3.0}), frozenset({0, 2, 3}), frozenset({1, 2, 3}), frozenset({0, 1, 2, 3})],
#             "expected_edges_[2, 3]": [(frozenset({2.0, 3.0}), frozenset({0, 2, 3})), (frozenset({2.0, 3.0}), frozenset({1, 2, 3})),
#                                       (frozenset({0, 2, 3}), frozenset({0, 1, 2, 3})), (frozenset({1, 2, 3}), frozenset({0, 1, 2, 3}))]
#         }


if __name__ == '__main__':
    unittest.main()
