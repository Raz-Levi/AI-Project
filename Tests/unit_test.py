"""
Automation Tests For The Project
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
import unittest
from utils import *
from sklearn import neighbors

from abstract_algorithm import LearningAlgorithm
from naive_algorithm import EmptyAlgorithm, RandomAlgorithm

""""""""""""""""""""""""""""""""""""""""" Tests  """""""""""""""""""""""""""""""""""""""""


class TestUtils(unittest.TestCase):
    # tests functions
    def test_get_samples_from_csv(self):
        consts = self._get_consts()
        matrix = get_samples_from_csv(consts["csv_path"])
        self.assertTrue(type(matrix) == np.ndarray)
        self.assertTrue(np.array_equal(matrix, consts["full_expected_matrix"]))

    def test_get_samples_from_csv_preprocessing(self):
        consts = self._get_consts()
        matrix = get_samples_from_csv(consts["csv_path"], self._pre_process_function_change_row)
        self.assertTrue(type(matrix) == np.ndarray)
        self.assertTrue(np.array_equal(matrix, consts["changed_row_expected_matrix"]))

        matrix = get_samples_from_csv(consts["csv_path"], self._pre_process_function_remove_first_row)
        self.assertTrue(type(matrix) == np.ndarray)
        self.assertTrue(np.array_equal(matrix, consts["removed_row_expected_matrix"]))

    def test_get_generator_for_samples_in_csv(self):
        consts = self._get_consts()
        matrix_generator = get_generator_for_samples_in_csv(consts["csv_path"])
        self._compare_generator_to_matrix(matrix_generator, consts["full_expected_matrix"])

    def test_get_generator_for_samples_in_csv_preprocessing_by_reference(self):
        consts = self._get_consts()
        matrix_generator = get_generator_for_samples_in_csv(consts["csv_path"], self._pre_process_function_change_row)
        self._compare_generator_to_matrix(matrix_generator, consts["changed_row_expected_matrix"])

        matrix_generator = get_generator_for_samples_in_csv(consts["csv_path"],
                                                            self._pre_process_function_remove_first_row)
        self._compare_generator_to_matrix(matrix_generator, consts["removed_row_expected_matrix"])

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

    # private functions
    @staticmethod
    def _get_consts():
        return {
            "csv_path": "test_csv_functions.csv",
            "full_expected_matrix": [[0.2, 0.11, 0.05], [1., 3.6, 5.4], [1., 2., 0.]],
            "changed_row_expected_matrix": [[0.2, 0.11, 0.05], [1., -3.6, -5.4], [1., -2., 0.]],
            "removed_row_expected_matrix": [[1., 3.6, 5.4], [1., 2., 0.]],
            "sample": np.array([[2, 2, 2]]),
            "classes": np.array([1]),
            "full_sample": np.array([[0, 1, 2, 3, 4, 5, 6]]),
            "given_features": np.array([2, 4, 6]),
            "given_features_not_sorted": np.array([6, 0, 2]),
            "given_features_full": np.array([5, 1, 2, 0, 6, 4, 3]),
            "total_features_num": 7,
            "default_value": 0,
            "completed_features_inf": np.array([[np.inf, np.inf, 2, np.inf, 2, np.inf, 2]]),
            "completed_features_zero": np.array([[0, 0, 2, 0, 2, 0, 2]]),
            "completed_features_not_sorted": np.array([[2, 0, 2, 0, 0, 0, 2]]),
            "completed_features_full": np.array([[3, 1, 2, 6, 5, 0, 4]])
        }

    def _compare_generator_to_matrix(self, matrix, expected_matrix):
        for row, expected_row in zip(matrix, expected_matrix):
            self.assertTrue(np.array_equal(row, expected_row))

    @staticmethod
    def _pre_process_function_change_row(row: np.ndarray):
        for i in range(len(row)):
            if row[i] > 1:
                row[i] = -row[i]

    @staticmethod
    def _pre_process_function_remove_first_row(row: np.ndarray):
        return [] if row[0] == 0.2 else row


class TestLearningAlgorithm(unittest.TestCase):
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
    def _get_consts():
        return {
            "test_sample": np.array([1]),
            "train_samples": TrainSamples(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]]), np.array([1, 1, 1, 1])),
            "given_feature": [1],
            "total_features_num": 3,
            "features_costs": [1],
            "maximal_cost": 1
        }

    @staticmethod
    def _get_instance():
        class SimpleAlgorithm(LearningAlgorithm):
            def _fit(self, train_samples: TrainSamples, features_costs: list[float]):
                pass

            def predict(self, sample: TestSamples, given_feature: list[int], maximal_cost: float) -> int:
                return True

            def _get_total_features_num(self):
                return self._total_features_num

        return SimpleAlgorithm()


class TestNaiveAlgorithm(unittest.TestCase):
    # tests functions
    def test_initializations(self):
        self.assertTrue(self._test_initialization(EmptyAlgorithm))
        # self.assertTrue(self._test_initialization(RandomAlgorithm))

    # private functions
    @staticmethod
    def _get_consts():
        return {
            "learning_algorithm": sklearn.neighbors.KNeighborsClassifier(n_neighbors=3),
        }

    @staticmethod
    def _test_initialization(tested_algorithm) -> bool:
        consts = TestNaiveAlgorithm._get_consts()
        algorithm = tested_algorithm(learning_algorithm=consts["learning_algorithm"])
        return type(algorithm) == EmptyAlgorithm and hasattr(algorithm.predict, '__call__') and hasattr(algorithm.fit, '__call__')

if __name__ == '__main__':
    unittest.main()
