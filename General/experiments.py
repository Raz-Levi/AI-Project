"""
Module for executing experiments
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from utils import *
from Tests.unit_test import get_features_cost_in_order
from General.score import ScoreFunctionA, ScoreFunctionB
from LearningAlgorithms.abstract_algorithm import LearningAlgorithm
from LearningAlgorithms.naive_algorithm import EmptyAlgorithm, RandomAlgorithm, OptimalAlgorithm
from LearningAlgorithms.mid_algorithm import MaxVarianceAlgorithm
from LearningAlgorithms.graph_search_algorithm import GraphSearchAlgorithm, LocalSearchAlgorithm
from LearningAlgorithms.genetic_algorithm import GeneticAlgorithm

import time
import sklearn.metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from networkx.algorithms.shortest_paths.astar import astar_path
from simpleai.search.local import hill_climbing, hill_climbing_stochastic, simulated_annealing, hill_climbing_random_restarts, beam

""""""""""""""""""""""""""""""""""""""""" Parameters """""""""""""""""""""""""""""""""""""""""

EXPERIMENTS_PARAMS = dict(
    datasets_path=[
        "../DataSets/WaterQualityDataSet/waterQuality_small.csv",
        "../DataSets/WaterQualityDataSet/waterQuality_medium.csv",
        "../DataSets/WaterQualityDataSet/waterQuality_big.csv"],
    class_index=20,
    train_ratio=None,
    features_costs=[],
    given_features=[2, 3, 6, 8, 11, 13, 16, 19],
    maximal_cost=20,
    random_seed=0,
    default_learning_algorithm=GraphSearchAlgorithm,
    default_classifier=LogisticRegression(),
    default_score_function=ScoreFunctionB(classifier=KNeighborsClassifier(1)),

    # hyperparameter_for_score_function_experiment
    n_split=5,
    random_state=0,
    cv_values=[i for i in range(1, 20)],

    # score_function_experiment
    score_functions=[ScoreFunctionA, ScoreFunctionB],
    parameters_for_score_functions={"classifier": KNeighborsClassifier(1), "alpha": 5},
    learning_algorithms_for_score_functions=[GraphSearchAlgorithm],

    # best_algorithms_experiment
    learning_algorithms=[EmptyAlgorithm, RandomAlgorithm, OptimalAlgorithm, MaxVarianceAlgorithm, GraphSearchAlgorithm, LocalSearchAlgorithm, GeneticAlgorithm],
    parameters_for_algorithms=[{"classifier": KNeighborsClassifier(3)},  # EmptyAlgorithm
                               {"classifier": KNeighborsClassifier(3)},  # RandomAlgorithm
                               {"classifier": KNeighborsClassifier(3)},  # OptimalAlgorithm
                               {"classifier": KNeighborsClassifier(3)},  # MaxVarianceAlgorithm
                               {"classifier": KNeighborsClassifier(3), "search_algorithm": astar_path, "score_function": ScoreFunctionB(classifier=KNeighborsClassifier(1))},  # GraphSearchAlgorithm
                               {"classifier": KNeighborsClassifier(3), "local_search_algorithm": hill_climbing, "score_function": ScoreFunctionB(classifier=KNeighborsClassifier(1))},  # LocalSearchAlgorithm
                               {"classifier": KNeighborsClassifier(3)}],  # GeneticAlgorithm

    # local_search_algorithm_experiment
    local_search_algorithms=[hill_climbing, hill_climbing_stochastic, simulated_annealing, hill_climbing_random_restarts, beam],
    parameters_for_local_search_algorithm=[{}, {"iterations_limit": 100}, {}, {"restarts_limit": 50}, {}],

    # best_classifier_experiment
    learning_algorithm_for_classifier_experiment=GeneticAlgorithm,
    classifiers=[KNeighborsClassifier, RandomForestClassifier, LogisticRegression],
    parameters_for_classifiers=[[1], [5], [5]]
)

GRAPHS_PARAMS = dict(
    # hyperparameter_for_score_function_experiment
    x_label_cv="Values",
    y_label_cv="Accuracy",

    # score_function_experiment
    x_values_score_function=["ScoreFunctionA", "ScoreFunctionB"],
    x_label_score_function="Score Function",
    y_label_score_function="Accuracy",

    # score_function_run_times_experiment
    x_values_score_function_run_times=["ScoreFunctionA", "ScoreFunctionB"],
    x_label_score_function_run_times="Score Function",
    y_label_score_function_run_times="Run Time",

    # best_algorithms_experiment
    x_values_best_algorithm=["Empty", "Random", "Optimal", "MaxVariance", "Graph", "Local", "Genetic"],
    x_label_best_algorithm="Algorithm",
    y_label_best_algorithm="Accuracy",

    # search_algorithm_experiment
    x_values_search_algorithm=["Hill Climbing", "Stochastic", "Annealing", "Restarts", "Beam"],
    x_label_search_algorithm="Search Algorithm",
    y_label_search_algorithm="Accuracy",

    # best_classifier_experiment
    x_values_best_classifier=[classifier.__name__ for classifier in EXPERIMENTS_PARAMS["classifiers"]],
    x_label_best_classifier="Classifier",
    y_label_best_classifier="Accuracy",

    # run_times_algorithms_experiment
    x_values_run_times=["Empty", "Random", "Optimal", "MaxVariance", "Graph", "Local" "Genetic"],
    x_label_run_times="Algorithm",
    y_label_run_times="Run Time",

    # run_times_features_algorithms_experiment
    x_values_run_times_features_num=["15", "20", "25", "30", "35", "40", "45", "50", "55"],
    x_label_run_times_features_num="number of features",
    y_label_run_times_features_num="Run Time",
)

""""""""""""""""""""""""""""""""""""""""""" Utils """""""""""""""""""""""""""""""""""""""""""


def get_accuracy(y_true: Union, y_pred: Union) -> float:
    return sklearn.metrics.accuracy_score(y_true, y_pred)


""""""""""""""""""""""""""""""""""""""""" Experiments """""""""""""""""""""""""""""""""""""""""


def execute_generic_experiment(train_samples: TrainSamples, test_samples: TestSamples, initialized_learning_algorithm: Type[LearningAlgorithm],
                               features_cost: Optional[List[int]] = None) -> float:
    features_cost = EXPERIMENTS_PARAMS["features_costs"] if features_cost is None else features_cost
    initialized_learning_algorithm.fit(train_samples=train_samples, features_costs=features_cost)
    y_pred = initialized_learning_algorithm.predict(samples=test_samples.samples,
                                                    given_features=EXPERIMENTS_PARAMS["given_features"],
                                                    maximal_cost=EXPERIMENTS_PARAMS["maximal_cost"])
    return get_accuracy(test_samples.classes, y_pred)


def hyperparameter_for_score_function_experiment(dataset_path: str):
    dataset, classes = get_samples_from_csv(dataset_path, 20)
    folds = KFold(n_splits=EXPERIMENTS_PARAMS["n_split"], shuffle=True, random_state=100)
    alpha_values, accuracies = EXPERIMENTS_PARAMS["cv_values"], []

    for alpha in alpha_values:
        accuracy = 0
        for train_fold, test_fold in folds.split(dataset):
            score_function = ScoreFunctionA(classifier=EXPERIMENTS_PARAMS["default_classifier"], alpha=alpha)
            learning_algorithm = EXPERIMENTS_PARAMS["default_learning_algorithm"](
                classifier=EXPERIMENTS_PARAMS["default_classifier"],
                score_function=score_function, search_algorithm=astar_path)
            t_samples = np.take(dataset, train_fold, 0)
            t_classes = np.take(classes, train_fold, 0)
            learning_algorithm.fit(TrainSamples(t_samples, t_classes), EXPERIMENTS_PARAMS["features_costs"])
            t_samples = np.take(dataset, test_fold, 0)
            test_samples = np.take(classes, test_fold, 0)
            y_pred = learning_algorithm.predict(t_samples, EXPERIMENTS_PARAMS["given_features"], EXPERIMENTS_PARAMS["maximal_cost"])
            accuracy += get_accuracy(test_samples, y_pred)
        accuracies.append(accuracy / EXPERIMENTS_PARAMS["n_split"])

    print_graph(alpha_values, accuracies, GRAPHS_PARAMS["x_label_cv"], GRAPHS_PARAMS["y_label_cv"])


def score_function_experiment(train_samples: TrainSamples, test_samples: TestSamples):
    accuracies = []
    for score_function_type, score_function_parameters in zip(EXPERIMENTS_PARAMS["score_functions"], EXPERIMENTS_PARAMS["parameters_for_score_functions"]):
        score_function = score_function_type(score_function_parameters)
        learning_algorithm = GraphSearchAlgorithm(KNeighborsClassifier(1), astar_path, score_function)
        accuracies.append(execute_generic_experiment(train_samples, test_samples, learning_algorithm))

    print_graph(GRAPHS_PARAMS["x_values_score_function"], accuracies, GRAPHS_PARAMS["x_label_score_function"], GRAPHS_PARAMS["y_label_score_function"])


def best_algorithms_experiment(train_samples: TrainSamples, test_samples: TestSamples):
    accuracies = []
    features_cost = get_features_cost_in_order(train_samples.get_features_num())
    for algorithm_type, algorithm_parameters in zip(EXPERIMENTS_PARAMS['learning_algorithms'], EXPERIMENTS_PARAMS['parameters_for_algorithms']):
        learning_algorithm = algorithm_type(**algorithm_parameters)
        accuracies.append(execute_generic_experiment(train_samples, test_samples, learning_algorithm, features_cost))

    print_graph(GRAPHS_PARAMS["x_values_best_algorithm"], accuracies, GRAPHS_PARAMS["x_label_best_algorithm"], GRAPHS_PARAMS["y_label_best_algorithm"])


def local_search_algorithm_experiment(train_samples: TrainSamples, test_samples: TestSamples):
    accuracies = []
    features_cost = get_features_cost_in_order(train_samples.get_features_num())
    for local_search_algorithm, local_search_algorithm_parameters in zip(EXPERIMENTS_PARAMS['local_search_algorithms'], EXPERIMENTS_PARAMS['parameters_for_local_search_algorithm']):
        score_function = EXPERIMENTS_PARAMS["default_score_function"]
        learning_algorithm = LocalSearchAlgorithm(EXPERIMENTS_PARAMS["default_classifier"], local_search_algorithm, score_function, **local_search_algorithm_parameters)
        accuracies.append(execute_generic_experiment(train_samples, test_samples, learning_algorithm, features_cost))

    print_graph(GRAPHS_PARAMS["x_values_search_algorithm"], accuracies, GRAPHS_PARAMS["x_label_search_algorithm"], GRAPHS_PARAMS["y_label_search_algorithm"])


def best_classifier_experiment(train_samples: TrainSamples, test_samples: TestSamples):
    accuracies = []
    classifieres = [KNeighborsClassifier(3), RandomForestClassifier(1), LogisticRegression()]
    for classifier in classifieres:
        learning_algorithm = EXPERIMENTS_PARAMS["learning_algorithm_for_classifier_experiment"](classifier=classifier)
        accuracies.append(execute_generic_experiment(train_samples, test_samples, learning_algorithm))

    print_graph(GRAPHS_PARAMS["x_values_best_classifier"], accuracies, GRAPHS_PARAMS["x_label_best_classifier"], GRAPHS_PARAMS["y_label_best_classifier"])


def run_time_experiment(train_samples: TrainSamples, test_samples: TestSamples):
    run_times = []
    features_cost = get_features_cost_in_order(train_samples.get_features_num())
    for algorithm_type, algorithm_parameters in zip(EXPERIMENTS_PARAMS['learning_algorithms'],
                                                    EXPERIMENTS_PARAMS['parameters_for_algorithms']):
        start = time.time()
        print(f'start {algorithm_type}')
        learning_algorithm = algorithm_type(**algorithm_parameters)
        execute_generic_experiment(train_samples, test_samples, learning_algorithm, features_cost)
        end = time.time()
        run_time = end - start
        run_times.append(run_time)

    print_graph(GRAPHS_PARAMS["x_values_run_times"], run_times, GRAPHS_PARAMS["x_label_run_times"],
                GRAPHS_PARAMS["y_label_run_times"])


def num_of_features_to_run_time(dataset_path : str):
    for algorithm_type, algorithm_parameters in zip(EXPERIMENTS_PARAMS['learning_algorithms'],
                                                    EXPERIMENTS_PARAMS['parameters_for_algorithms']):
        run_times = []
        for d in range(15, 60, 5):
            train_samples, test_samples = get_dataset_with_num_of_features(d, dataset_path, class_index=EXPERIMENTS_PARAMS["class_index"],
                                                      train_ratio=EXPERIMENTS_PARAMS["train_ratio"], random_seed=0,
                                                      shuffle=True)
            features_cost = get_features_cost_in_order(train_samples.get_features_num())
            start = time.time()
            learning_algorithm = algorithm_type(**algorithm_parameters)
            execute_generic_experiment(train_samples, test_samples, learning_algorithm, features_cost)
            end = time.time()
            run_time = end - start
            run_times.append(run_time)

        print_graph(GRAPHS_PARAMS["x_values_run_times_features_num"], run_times, GRAPHS_PARAMS["x_label_run_times_features_num"],
                GRAPHS_PARAMS["y_label_run_times_features_num"])


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def execute_experiments():
    for dataset_path in EXPERIMENTS_PARAMS["datasets_path"]:
        train_samples, test_samples = get_dataset(dataset_path, class_index=EXPERIMENTS_PARAMS["class_index"],
                                                  train_ratio=EXPERIMENTS_PARAMS["train_ratio"], random_seed=0, shuffle=True)
        hyperparameter_for_score_function_experiment(dataset_path)
        score_function_experiment(train_samples, test_samples)
        best_algorithms_experiment(train_samples, test_samples)
        local_search_algorithm_experiment(train_samples, test_samples)
        best_classifier_experiment(train_samples, test_samples)
        run_time_experiment(train_samples, test_samples)
        num_of_features_to_run_time(dataset_path)


if __name__ == '__main__':
    execute_experiments()
