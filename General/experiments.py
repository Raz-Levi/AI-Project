"""
Module for executing experiments
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""

from utils import *
from General.score import ScoreFunctionA, ScoreFunctionB
from LearningAlgorithms.abstract_algorithm import LearningAlgorithm
from LearningAlgorithms.naive_algorithm import EmptyAlgorithm, RandomAlgorithm, OptimalAlgorithm
from LearningAlgorithms.mid_algorithm import MaxVarianceAlgorithm
from LearningAlgorithms.graph_search_algorithm import GraphSearchAlgorithm

import sklearn.metrics
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from networkx.algorithms.shortest_paths.astar import astar_path
from networkx.algorithms.shortest_paths.generic import shortest_path

""""""""""""""""""""""""""""""""""""""""" Parameters """""""""""""""""""""""""""""""""""""""""

EXPERIMENTS_PARAMS = dict(  # TODO: update the parameters after we are deciding them
    datasets_path=[],
    train_ratio=1,
    features_costs=[],
    given_features=[],
    maximal_cost=0,
    default_learning_algorithm=GraphSearchAlgorithm,  # TODO: decide default
    default_classifier=KNeighborsClassifier,
    default_score_function=ScoreFunctionB(classifier=KNeighborsClassifier),

    # hyperparameter_for_score_function_experiment
    n_split=5,
    random_state=0,
    cv_values=[],  # TODO: decide on values

    # score_function_experiment
    score_functions=[ScoreFunctionA, ScoreFunctionB],
    parameters_for_score_functions=[],  # TODO: here we should insert parameters for the algorithms according their order in classifiers list
    learning_algorithms_for_score_functions=[GraphSearchAlgorithm],
    learning_algorithms_param_for_score_functions=[],  # TODO: here we should insert parameters for the algorithms according their order in classifiers list

    # best_algorithms_experiment
    learning_algorithms=[EmptyAlgorithm, RandomAlgorithm, OptimalAlgorithm, MaxVarianceAlgorithm, GraphSearchAlgorithm],  # TODO: add genetic algo
    parameters_for_algorithms=[],  # TODO: here we should insert parameters for the algorithms according their order in classifiers list

    # search_algorithm_experiment
    search_algorithms=[astar_path, shortest_path],
    parameters_for_search_algorithm=[None, "dijkstra", "bellman-ford"],  # TODO: here we should insert parameters for the algorithms according their order in classifiers list

    # best_classifier_experiment
    learning_algorithm_for_classifier_experiment=GraphSearchAlgorithm,  # TODO: think about it
    classifiers=[MLPClassifier, KNeighborsClassifier, RandomForestClassifier, LogisticRegression],
    parameters_for_classifiers=[]  # TODO: here we should insert parameters for the classifiers according their order in classifiers list

    # MLP: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    # KNN: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    # RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    # LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
)

GRAPHS_PARAMS = dict(
    # hyperparameter_for_score_function_experiment
    x_label_cv="Values",
    y_label_cv="Accuracy",

    # score_function_experiment
    x_values_score_function=["ScoreFunctionA", "ScoreFunctionB"],
    x_label_score_function="Score Function",
    y_label_score_function="Accuracy",

    # best_algorithms_experiment
    x_values_best_algorithm=["EmptyAlgorithm", "RandomAlgorithm", "OptimalAlgorithm", "MaxVarianceAlgorithm", "GraphSearchAlgorithm"],  # TODO: add genetic algo
    x_label_best_algorithm="Algorithm",
    y_label_best_algorithm="Accuracy",

    # search_algorithm_experiment
    x_values_search_algorithm=["Astar", "Dijkstra", "Bellman Ford"],  # TODO: update
    x_label_search_algorithm="Search Algorithm",
    y_label_search_algorithm="Accuracy",

    # best_classifier_experiment
    x_values_best_classifier=[classifier.__name__ for classifier in EXPERIMENTS_PARAMS["classifiers"]],
    x_label_best_classifier="Classifier",
    y_label_best_classifier="Accuracy"
)

""""""""""""""""""""""""""""""""""""""""""" Utils """""""""""""""""""""""""""""""""""""""""""


def get_accuracy(y_true: Union, y_pred: Union) -> float:
    return sklearn.metrics.accuracy_score(y_true, y_pred)


""""""""""""""""""""""""""""""""""""""""" Experiments """""""""""""""""""""""""""""""""""""""""


def execute_generic_experiment(train_samples: TrainSamples, test_samples: TestSamples, initialized_learning_algorithm: Type[LearningAlgorithm]) -> float:
    initialized_learning_algorithm.fit(train_samples=train_samples, features_costs=EXPERIMENTS_PARAMS["features_costs"])
    y_pred = initialized_learning_algorithm.predict(samples=test_samples,
                                                    given_features=EXPERIMENTS_PARAMS["given_features"],
                                                    maximal_cost=EXPERIMENTS_PARAMS["maximal_cost"])
    return get_accuracy(test_samples.classes, y_pred)


def hyperparameter_for_score_function_experiment(train_samples: TrainSamples):
    folds = KFold(n_splits=EXPERIMENTS_PARAMS["n_split"], shuffle=True, random_state=["random_state"])
    alpha_values, accuracies = EXPERIMENTS_PARAMS["cv_values"], []

    for alpha in alpha_values:
        accuracy = 0
        for train_fold, test_fold in folds.split(train_samples):
            score_function = ScoreFunctionA(classifier=EXPERIMENTS_PARAMS["default_classifier"], alpha=alpha)
            learning_algorithm = EXPERIMENTS_PARAMS["default_learning_algorithm"](learning_algorithm=EXPERIMENTS_PARAMS["default_classifier"], score_function=score_function)  # TODO: add params
            learning_algorithm.fit(np.take(train_samples, train_fold, 0), EXPERIMENTS_PARAMS["features_costs"])
            accuracy += learning_algorithm.predict(np.take(train_samples, test_fold, 0), EXPERIMENTS_PARAMS["given_features"], EXPERIMENTS_PARAMS["maximal_cost"])
        accuracies.append(accuracy / EXPERIMENTS_PARAMS["n_split"])

    print_graph(alpha_values, accuracies, GRAPHS_PARAMS["x_label_cv"], GRAPHS_PARAMS["y_label_cv"])
    # return alpha_values[int(np.argmax(accuracies))] # TODO: remove it after get the best param


def score_function_experiment(train_samples: TrainSamples, test_samples: TestSamples):
    accuracies = []
    for learning_algorithm_type, learning_algorithm_param in zip(EXPERIMENTS_PARAMS["learning_algorithms_for_score_functions"], EXPERIMENTS_PARAMS["learning_algorithms_param_for_score_functions"]):
        for score_function_type, score_function_parameters in zip(EXPERIMENTS_PARAMS["score_functions"], EXPERIMENTS_PARAMS["parameters_for_score_functions"]):
            score_function = score_function_type(score_function_parameters)  # TODO: during experiments, verify the parameters for init function
            learning_algorithm = learning_algorithm_type(learning_algorithm_param, score_function=score_function)  # TODO: during experiments, verify the parameters for init function
            accuracies.append(execute_generic_experiment(train_samples, test_samples, learning_algorithm))

        print_graph(GRAPHS_PARAMS["x_values_score_function"], accuracies, GRAPHS_PARAMS["x_label_score_function"], GRAPHS_PARAMS["y_label_score_function"])


def best_algorithms_experiment(train_samples: TrainSamples, test_samples: TestSamples):
    accuracies = []
    for algorithm_type, algorithm_parameters in zip(EXPERIMENTS_PARAMS['learning_algorithms'], EXPERIMENTS_PARAMS['parameters_for_algorithms']):
        learning_algorithm = algorithm_type(algorithm_parameters)  # TODO: during experiments, verify the parameters for init function
        accuracies.append(execute_generic_experiment(train_samples, test_samples, learning_algorithm))

    print_graph(GRAPHS_PARAMS["x_values_best_algorithm"], accuracies, GRAPHS_PARAMS["x_label_best_algorithm"], GRAPHS_PARAMS["y_label_best_algorithm"])


def search_algorithm_experiment(train_samples: TrainSamples, test_samples: TestSamples):
    accuracies = []
    for search_algorithm, search_algorithm_parameters in zip(EXPERIMENTS_PARAMS['search_algorithms'], EXPERIMENTS_PARAMS['parameters_for_search_algorithm']):
        score_function = EXPERIMENTS_PARAMS["default_score_function"]
        learning_algorithm = GraphSearchAlgorithm(EXPERIMENTS_PARAMS["default_classifier"], search_algorithm, score_function, search_algorithm_parameters)  # TODO: during experiments, verify astar is running
        accuracies.append(execute_generic_experiment(train_samples, test_samples, learning_algorithm))

    print_graph(GRAPHS_PARAMS["x_values_search_algorithm"], accuracies, GRAPHS_PARAMS["x_label_search_algorithm"], GRAPHS_PARAMS["y_label_search_algorithm"])


def best_classifier_experiment(train_samples: TrainSamples, test_samples: TestSamples):
    accuracies = []
    for classifier_type, classifier_parameters in zip(EXPERIMENTS_PARAMS['classifiers'], EXPERIMENTS_PARAMS["parameters_for_classifiers"]):
        classifier = classifier_type(classifier_parameters)
        learning_algorithm = EXPERIMENTS_PARAMS["learning_algorithm_for_classifier_experiment"](classifier=classifier)  # TODO: during experiments, verify the parameters for init function
        accuracies.append(execute_generic_experiment(train_samples, test_samples, learning_algorithm))

    print_graph(GRAPHS_PARAMS["x_values_best_classifier"], accuracies, GRAPHS_PARAMS["x_label_best_classifier"], GRAPHS_PARAMS["y_label_best_classifier"])


""""""""""""""""""""""""""""""""""""""""""" Main """""""""""""""""""""""""""""""""""""""""""


def execute_experiments():
    for dataset_path in EXPERIMENTS_PARAMS["datasets_path"]:
        train_samples, test_samples = get_dataset(dataset_path, train_ratio=EXPERIMENTS_PARAMS["train_ratio"])

        hyperparameter_for_score_function_experiment(train_samples)
        score_function_experiment(train_samples, test_samples)
        best_algorithms_experiment(train_samples, test_samples)
        search_algorithm_experiment(train_samples, test_samples)
        best_classifier_experiment(train_samples, test_samples)


if __name__ == '__main__':
    execute_experiments()
