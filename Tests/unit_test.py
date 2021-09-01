"""
Automation Tests For The Project
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
import unittest
from tests_parameters import *
from networkx.algorithms.shortest_paths.astar import astar_path
from networkx.algorithms.shortest_paths.generic import shortest_path
from simpleai.search.local import hill_climbing

from General.score import ScoreFunction, ScoreFunctionA, ScoreFunctionB
from LearningAlgorithms.abstract_algorithm import LearningAlgorithm
from LearningAlgorithms.naive_algorithm import EmptyAlgorithm, RandomAlgorithm, OptimalAlgorithm
from LearningAlgorithms.mid_algorithm import MaxVarianceAlgorithm
from LearningAlgorithms.graph_search_algorithm import GraphSearchAlgorithm, LocalSearchAlgorithm
from LearningAlgorithms.genetic_algorithm import GeneticAlgorithm

""""""""""""""""""""""""""""""""""""""""" Utils  """""""""""""""""""""""""""""""""""""""""


def get_features_cost_in_order(features_num: int) -> List[int]:
    return list(range(1, features_num + 1))


""""""""""""""""""""""""""""""""""""""""" Tests  """""""""""""""""""""""""""""""""""""""""


class TestUtils(unittest.TestCase):
    # tests functions
    def test_get_samples_from_csv(self):
        self._test_get_samples_from_csv(path=NUMERIC_SAMPLES_PATH, expected_matrix=NUMERIC_SAMPLES_EXPECTED)

    def test_categorical_to_numeric(self):
        categories = {}
        self._test_get_samples_from_csv(path=STRING_SAMPLES_PATH,
                                        expected_matrix=STRING_SAMPLES_EXPECTED,
                                        preprocess=categorical_to_numeric,
                                        categories=categories)
        categories = {}
        self._test_get_samples_from_csv(path=FEW_REAL_SAMPLES_PATH,
                                        expected_matrix=FEW_REAL_SAMPLES_EXPECTED,
                                        preprocess=categorical_to_numeric,
                                        categories=categories)

    def test_get_dataset(self):
        for ratio in TRAIN_RATIO_BATCH:
            self._test_get_dataset(path=NUMERIC_SAMPLES_PATH, expected_matrix=NUMERIC_SAMPLES_EXPECTED, train_ratio=ratio, random_seed=RANDOM_SEED)
            self._test_get_dataset(path=STRING_SAMPLES_PATH, expected_matrix=STRING_SAMPLES_EXPECTED, train_ratio=ratio, random_seed=RANDOM_SEED)

    def test_declarations(self):
        train_samples = TrainSamples(TRAIN_SAMPLE, TRAIN_CLASSES)
        self.assertTrue(np.array_equal(train_samples.samples, TRAIN_SAMPLE))
        self.assertTrue(np.array_equal(train_samples.classes, TRAIN_CLASSES))

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
            complementary_list = list(get_complementary_set(range(train_samples.get_samples_num()), tested_rows))
            self.assertTrue(self._compare_samples(train_samples.samples, train_samples.classes, expected_matrix[complementary_list, :], col))
            self.assertTrue(self._compare_samples(test_samples.samples, test_samples.classes, expected_matrix[tested_rows, :], col))

    @staticmethod
    def _compare_samples(samples: np.array, classes: np.array, expected_matrix: np.array, class_index: int, **kw) -> bool:
        complementary_list = list(get_complementary_set(range(expected_matrix.shape[1]), [class_index]))
        expected_samples, expected_classes = expected_matrix[:, complementary_list], expected_matrix[:, [class_index]]
        return type(samples) == np.ndarray and np.array_equal(samples, expected_samples) and type(
            classes) == np.ndarray and np.array_equal(classes, expected_classes.flatten())


class TestLearningAlgorithm(unittest.TestCase):
    # tests functions
    def test_initialization(self):
        simple_algorithm = self._get_instance()
        self.assertTrue(simple_algorithm.predict(TEST_SAMPLE, GIVEN_FEATURE_ONE, MAXIMAL_COST_LOW))

    def test_fit(self):
        simple_algorithm = self._get_instance()
        self.assertTrue(simple_algorithm._get_total_features_num() is None)
        simple_algorithm.fit(TRAIN_SAMPLES_BIG, get_features_cost_in_order(TRAIN_SAMPLES_BIG.get_features_num()))
        self.assertEqual(simple_algorithm._get_total_features_num(), TRAIN_SAMPLES_BIG.get_features_num())

    # private functions
    @staticmethod
    def _get_instance() -> Type[LearningAlgorithm]:
        class SimpleAlgorithm(LearningAlgorithm):
            def __init__(self):
                self._total_features_num = None

            def fit(self, train_samples: TrainSamples, features_costs: list[float]):
                self._total_features_num = train_samples.get_features_num()

            def predict(self, sample: TestSamples, given_feature: GivenFeatures, maximal_cost: float) -> int:
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
    def _test_initialization(tested_algorithm) -> bool:
        algorithm = tested_algorithm(classifier=CLASSIFIER)
        return type(algorithm) == tested_algorithm and hasattr(algorithm.predict, '__call__') and hasattr(algorithm.fit, '__call__')

    @staticmethod
    def _test_naive_algorithm(tested_algorithm) -> Tuple[bool, Type[LearningAlgorithm]]:
        algorithm = tested_algorithm(classifier=CLASSIFIER)
        test_result = type(algorithm) == tested_algorithm
        algorithm.fit(train_samples=TRAIN_SAMPLES_BIG, features_costs=get_features_cost_in_order(TRAIN_SAMPLES_BIG.get_features_num()))

        predicted_sample = algorithm.predict(samples=TRAIN_SAMPLES_BIG.samples,
                                             given_features=GIVEN_FEATURES,
                                             maximal_cost=MAXIMAL_COST_LOW)
        test_result = test_result and np.array_equal(predicted_sample, TRAIN_SAMPLES_BIG.classes)

        predicted_sample = algorithm.predict(samples=np.array([TRAIN_SAMPLES_BIG.samples[0]]),
                                             given_features=GIVEN_FEATURES_MISSED,
                                             maximal_cost=MAXIMAL_COST_LOW)
        test_result = test_result and np.array_equal(predicted_sample.item(), TRAIN_SAMPLES_BIG.classes[0])

        predicted_sample = algorithm.predict(samples=TRAIN_SAMPLES_BIG.samples,
                                             given_features=GIVEN_FEATURES_MISSED,
                                             maximal_cost=MAXIMAL_COST_LOW)
        test_result = test_result and np.array_equal(predicted_sample, TRAIN_SAMPLES_BIG.classes)
        return test_result, algorithm

    @staticmethod
    def _test_sequence_algorithm(tested_algorithm) -> Tuple[bool, Type[LearningAlgorithm]]:
        test_result, algorithm = TestNaiveAlgorithm._test_naive_algorithm(tested_algorithm)
        predicted_sample = algorithm.predict(samples=TRAIN_SAMPLES_BIG.samples,
                                             given_features=GIVEN_FEATURE_EMPTY,
                                             maximal_cost=MAXIMAL_COST_PARTIALLY)
        test_result = test_result and np.array_equal(predicted_sample, TRAIN_SAMPLES_BIG.classes)
        return test_result, algorithm

    @staticmethod
    def _test_mid_algorithm(tested_algorithm) -> Tuple[bool, Type[LearningAlgorithm]]:
        test_result, algorithm = TestNaiveAlgorithm._test_naive_algorithm(tested_algorithm)
        predicted_sample = algorithm.predict(samples=TRAIN_SAMPLES_BIG.samples,
                                             given_features=GIVEN_FEATURES_MISSED,
                                             maximal_cost=MAXIMAL_COST_PARTIALLY)
        test_result = test_result and np.array_equal(predicted_sample, TRAIN_SAMPLES_BIG.classes)
        return test_result, algorithm


class TestScoreFunction(unittest.TestCase):
    # tests functions
    def test_function_scoreA_1(self):
        score_function = ScoreFunctionA(alpha=ALPHA_ONE)
        self.assertEqual(A_ALPHA_ONE_RESULT, score_function(train_samples=TrainSamples(CORR_MATRIX, CORR_CLASSES),
                                                            given_features=GIVEN_FEATURES_FOR_SCORE_TEST_ALPHA_ONE,
                                                            new_feature=NEW_FEATURE_ONE,
                                                            costs_list=FEATURES_COST_IN_ORDER))

    def test_function_scoreA_2(self):
        score_function = ScoreFunctionA(alpha=ALPHA_TWO)
        self.assertEqual(A_ALPHA_TWO_RESULT, score_function(train_samples=TrainSamples(CORR_MATRIX, CORR_CLASSES),
                                                            given_features=GIVEN_FEATURES_FOR_SCORE_TEST_ALPHA_TWO,
                                                            new_feature=NEW_FEATURE_ONE,
                                                            costs_list=FEATURES_COST_IN_ORDER))

    def test_function_scoreB_1(self):
        score_function = ScoreFunctionB(classifier=CLASSIFIER, alpha=ALPHA_ONE)
        self.assertEqual(B_ALPHA_ONE_RESULT, score_function(train_samples=TrainSamples(CORR_MATRIX, CORR_CLASSES),
                                                            given_features=GIVEN_FEATURES_FOR_SCORE_TEST_ALPHA_ONE,
                                                            new_feature=NEW_FEATURE_ONE,
                                                            costs_list=FEATURES_COST_IN_ORDER))

    def test_function_scoreB_2(self):
        score_function = ScoreFunctionB(classifier=CLASSIFIER, alpha=ALPHA_TWO)
        self.assertEqual(B_ALPHA_TWO_RESULT, score_function(train_samples=TrainSamples(CORR_MATRIX, CORR_CLASSES),
                                                            given_features=GIVEN_FEATURES_FOR_SCORE_TEST_ALPHA_TWO,
                                                            new_feature=NEW_FEATURE_TWO,
                                                            costs_list=FEATURES_COST_IN_ORDER))


class TestGraphSearchAlgorithm(unittest.TestCase):
    # tests functions
    def test_initialization(self):
        self.assertTrue(self._test_initialization(GraphSearchAlgorithm))
        self.assertTrue(self._test_initialization(LocalSearchAlgorithm))

    def test_build_graph(self):
        algorithm = self._get_algorithm_instance(GraphSearchAlgorithm)
        for given_features in GIVEN_FEATURES_BATCH:
            features_costs = [i for i in range(len(get_complementary_set(range(TRAIN_SAMPLES_SMALL.get_features_num()), given_features)))]
            algorithm.fit(TRAIN_SAMPLES_SMALL, features_costs)
            algorithm._build_graph(total_features=TRAIN_SAMPLES_SMALL.get_features_num(), given_features=given_features)
            self.assertEqual(list(algorithm._graph.nodes), EXPECTED_NODES[f'expected_nodes_{given_features}'])
            self.assertEqual(list(algorithm._graph.edges), EXPECTED_NODES[f'expected_edges_{given_features}'])
            self.assertEqual(list(algorithm._graph.nodes)[0], frozenset(given_features))
            self.assertEqual(list(algorithm._graph.nodes)[-1], frozenset(range(TRAIN_SAMPLES_SMALL.get_features_num())))

    def test_features_costs_heuristic(self):
        algorithm = self._get_algorithm_instance(GraphSearchAlgorithm)
        algorithm.fit(TRAIN_SAMPLES_SMALL, list(range(1, TRAIN_SAMPLES_SMALL.get_features_num() + 1)))
        for tested_nodes in FEATURES_COST_HEURISTIC_EXPECTED:
            self.assertEqual(algorithm._features_costs_heuristic(tested_nodes[0], tested_nodes[1]), tested_nodes[2])

    def test_buy_features(self):
        train_samples, _ = get_dataset(NUMERIC_SAMPLES_PATH, TRAIN_RATIO_BATCH[0])
        features_costs = get_features_cost_in_order(train_samples.get_features_num())
        algorithm = self._get_algorithm_instance(GraphSearchAlgorithm)
        algorithm.fit(train_samples, features_costs)
        for given_features in GIVEN_FEATURES_BATCH:
            new_given_features = algorithm._buy_features(given_features[:], MAXIMAL_COST_HIGH)
            self.assertEqual(sorted(new_given_features), list(range(train_samples.get_features_num())))
        new_given_features = algorithm._buy_features(GIVEN_FEATURES_BATCH[0], MAXIMAL_COST_LOW)
        self.assertEqual(new_given_features, GIVEN_FEATURES_FOR_SMALL_COST)

    def test_graph_search_algorithm(self):
        simple_algorithm = self._get_algorithm_instance(GraphSearchAlgorithm)
        score_function_algorithm = self._get_algorithm_instance(GraphSearchAlgorithm, score_function_type=ScoreFunctionB)
        dijkstra_algorithm = self._get_algorithm_instance(GraphSearchAlgorithm, search_algorithm=shortest_path)
        self._full_classification_test(simple_algorithm)
        self._full_classification_test(score_function_algorithm)
        self._full_classification_test(dijkstra_algorithm)

    def test_get_best_state(self):
        for train_ratio in TRAIN_RATIO_BATCH:
            train_samples, _ = get_dataset(NUMERIC_SAMPLES_PATH, train_ratio=train_ratio)
            algorithm = self._get_algorithm_instance(LocalSearchAlgorithm, self._get_depth_score_function())
            features_costs = get_features_cost_in_order(train_samples.get_features_num())
            algorithm.fit(train_samples, features_costs)
            best_state = algorithm._get_best_state(GIVEN_FEATURES_BATCH[0], MAXIMAL_COST_HIGH)
            self.assertEqual(sorted(best_state), BEST_STATE_EXPECTED)

    def test_local_search_algorithm(self):
        local_search_algorithm = self._get_algorithm_instance(LocalSearchAlgorithm, self._get_depth_score_function())
        self._full_classification_test(local_search_algorithm)

    # private functions
    def _test_initialization(self, algorithm_type: Type[LearningAlgorithm]):
        algorithm = self._get_algorithm_instance(algorithm_type)
        return type(algorithm) == algorithm_type and hasattr(algorithm.predict, '__call__') and hasattr(algorithm.fit, '__call__')

    def _full_classification_test(self, algorithm: Type[LearningAlgorithm]):
        for train_ratio in TRAIN_RATIO_BATCH:
            train_samples, _ = get_dataset(NUMERIC_SAMPLES_PATH, train_ratio=train_ratio)
            features_costs = get_features_cost_in_order(train_samples.get_features_num())
            algorithm.fit(train_samples, features_costs)
            for given_features in GIVEN_FEATURES_BATCH:
                predicted_classes = algorithm.predict(train_samples.samples, given_features, MAXIMAL_COST_HIGH)
                self.assertTrue(np.array_equal(predicted_classes, train_samples.classes))

    def _get_algorithm_instance(self, algorithm_type: Type[LearningAlgorithm], score_function_type: Optional[Type[ScoreFunction]] = None,
                                search_algorithm: nx.algorithms = astar_path, local_search_algorithm: Optional[Callable] = hill_climbing) -> Type[LearningAlgorithm]:
        score_function_type = self._get_simple_score_function() if score_function_type is None else score_function_type
        score_function = score_function_type(classifier=CLASSIFIER)
        if algorithm_type == GraphSearchAlgorithm:
            return GraphSearchAlgorithm(CLASSIFIER, search_algorithm, score_function)
        else:
            return LocalSearchAlgorithm(CLASSIFIER, local_search_algorithm, score_function)

    @staticmethod
    def _get_depth_score_function():
        class DepthScoreFunction(ScoreFunction):
            def _execute_function(self, train_samples: TrainSamples, given_features: GivenFeatures, new_feature: int, costs_list: list[float]) -> float:
                return len(given_features) + 1
        return DepthScoreFunction

    @staticmethod
    def _get_simple_score_function():
        class SimpleScore(ScoreFunction):
            def _execute_function(self, train_samples: TrainSamples, given_features: GivenFeatures, new_feature: int, costs_list: list[float]) -> float:
                return 0.2
        return SimpleScore


class TestGeneticAlgorithm(unittest.TestCase):
    # tests functions
    def test_initialization(self):
        algorithm = GeneticAlgorithm(classifier=CLASSIFIER)
        return type(algorithm) == GeneticAlgorithm

    def test_buy_features(self):
        algorithm = GeneticAlgorithm(classifier=CLASSIFIER)
        train_samples, _ = get_dataset(HEART_FAILURE_SAMPLES_PATH, train_ratio=TRAIN_RATIO, class_index=CLASS_INDEX)
        algorithm.fit(train_samples, FEATURES_COST_LARGE)
        res = algorithm._buy_features(GIVEN_FEATURES_BATCH[0], MAXIMAL_COST_LOW)
        self.assertTrue(algorithm._is_legal_subset(res))


if __name__ == '__main__':
    unittest.main()
