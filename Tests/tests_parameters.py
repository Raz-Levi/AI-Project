"""
Parameters for the unit test
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from General.utils import *
from sklearn.neighbors import KNeighborsClassifier

""""""""""""""""""""""""""""""""""""""""""" Parameters """""""""""""""""""""""""""""""""""""""""""

# Path for csv files that contains dataset for example
NUMERIC_SAMPLES_PATH = "TestDataSets/test_csv_functions.csv"
STRING_SAMPLES_PATH = "TestDataSets/test_csv_with_strings.csv"
FEW_REAL_SAMPLES_PATH = "TestDataSets/test_csv_few_samples.csv"
HEART_FAILURE_SAMPLES_PATH = "TestDataSets/heart_failure_clinical_records_dataset.csv"

# Samples for example
TRAIN_SAMPLE = np.array([[2, 2, 2]])
TRAIN_CLASSES = np.array([1])
TRAIN_SAMPLES_SMALL = TrainSamples(train_samples=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), train_classes=np.array([[1, 0]]))
TRAIN_SAMPLES_BIG = TrainSamples(train_samples=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]]), train_classes=np.array([1, 1, 1, 1]))
TEST_SAMPLE = np.array([1])

# Algorithms parameters
CLASSIFIER = KNeighborsClassifier(n_neighbors=1)
GIVEN_FEATURE_EMPTY = []
GIVEN_FEATURE_ONE = [1]
GIVEN_FEATURES = [0, 1, 2]
GIVEN_FEATURES_FOR_SCORE_TEST_ALPHA_ONE = [2]
GIVEN_FEATURES_FOR_SCORE_TEST_ALPHA_TWO = [0, 2]
GIVEN_FEATURES_MISSED = [0, 1]
GIVEN_FEATURES_BATCH = [[0], [3], [2, 3]]
FEATURES_COST_IN_ORDER = [1, 2, 3]
FEATURES_COST_LARGE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 22, 23]
MAXIMAL_COST_LOW = 10
MAXIMAL_COST_HIGH = 1000
MAXIMAL_COST_PARTIALLY = 4
CONSIDERED_FEATURES_NUM = 6

# Dataset parameters
RANDOM_SEED = 0
TRAIN_RATIO = 1
TRAIN_RATIO_BATCH = [1, 2, 3, 4]
CLASS_INDEX = 12

# Score function
SCORE_RESULT = 0.07012591041294361
CORR_MATRIX = [[1, 2, 3], [-2, -4, -6], [3, 6, -5]]
CORR_CLASSES = [0, 0, 1]
NEW_FEATURE_ONE = 1
NEW_FEATURE_TWO = 2
ALPHA_ONE = 1
A_ALPHA_ONE_RESULT = 0
B_ALPHA_ONE_RESULT = 0.5
ALPHA_TWO = 2
A_ALPHA_TWO_RESULT = 21
B_ALPHA_TWO_RESULT = 1/3

# Local Search
BEST_STATE_EXPECTED = [0, 1, 2, 3, 4, 5]

# Expected dataset
NUMERIC_SAMPLES_EXPECTED = np.array([[1, 0.11, 0.05, 78, 32, 12, 4231],
                                    [0, 3.6, 5.4, 4.32, 432.2, 21.4, 43.21],
                                    [1, 2, 0, 43, 21, 245, 4.231],
                                    [1, 22, 32, 6, 3.45, 62.4, 2.2],
                                    [62, 32, 12, 214, 215, 53.215, 21]])

STRING_SAMPLES_EXPECTED = np.array([[0.2, 0., 0.05, 0., 0., 0., 0.9, 0, 0, 3],
                                    [1., 0., 5.4, 1., 0., 1., 1.2, 1, 1, 10],
                                    [1., 1., 0., 0., 1., 1., 5.6, 0, 1, 20],
                                    [0.1, 1, 0.4, 0., 1., 0., 0., 1, 0, 10],
                                    [0.3, 0, 0.9, 1, 1, 0, 1.8, 1, 1, 3],
                                    [0.4, 0, 1.2, 1, 0, 1, 0.3, 0, 0, 20]])

FEW_REAL_SAMPLES_EXPECTED = np.array([[0, 0, 0, 13, 0, 0, 460, 3, 4, 0],
                                      [1, 0, 1, 25, 1, 1, 235, 3, 2, 0],
                                      [2, 1, 0, 26, 1, 1, 1142, 2, 2, 1]])
