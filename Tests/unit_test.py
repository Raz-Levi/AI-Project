"""
Automation Tests For The Project
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
import unittest
import typing
from utils import *
from sklearn.neighbors import KNeighborsClassifier

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
        self.assertEqual(1/2, res)
        res = score_function_a(data, [1, 2], 3, 0, costs_list, alpha=2)
        self.assertEqual(0.5502954390354358, res)

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
    @staticmethod
    def _get_consts():
        return {
            "csv_path": "test_csv_functions.csv",
            "full_expected_matrix": [[0.2, 0.11, 0.05], [1., 3.6, 5.4], [1., 2., 0.]],
            "changed_row_expected_matrix": [[0.2, 0.11, 0.05], [1., -3.6, -5.4], [1., -2., 0.]],
            "removed_row_expected_matrix": [[1., 3.6, 5.4], [1., 2., 0.]],
            "corr_matrix": [[1, 2, 3, 0], [-2, -4, -6, 0], [3, 6, -5, 1]],
            "costs_list": [1, 2, 3, 4]
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


if __name__ == '__main__':
    unittest.main()
