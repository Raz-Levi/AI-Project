"""
Utils For The Project
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
import scipy.stats as stats
import sklearn

import abc
from typing import Callable, Iterator
from dataclasses import dataclass

""""""""""""""""""""""""""""""""""" Definitions and Consts """""""""""""""""""""""""""""""""""

Sample = np.array
TestSamples = np.array
Classes = np.array


@dataclass
class TrainSamples:
    samples: np.array  # Training data- shape (n_samples, n_features), i.e samples are in the rows.
    classes: Classes  # Target values- shape (n_samples,).


""""""""""""""""""""""""""""""""""""""""""" Methods """""""""""""""""""""""""""""""""""""""""""


def get_samples_from_csv(path: str, preprocess: Callable = None) -> np.array:
    """
    get samples from csv as np.ndarray. notice that the first row (titles) are being ignored.
    :param path: string that contains the path for the csv.
    :param preprocess: (optional) function for preprocess the data. the function can change the values by reference,
                        can change by value (the function should return specific sample), or can remove sample due to a
                        condition (the function should return []).
    :return: samples as np.ndarray.
    """
    samples, data_frame = [], pd.read_csv(filepath_or_buffer=path, sep=",")
    for example in data_frame.values:
        sample = list(example)
        if preprocess is not None:
            processed_sample = preprocess(sample)
            sample = sample if processed_sample is None else processed_sample
        if sample:
            samples.append(sample)
    return np.array(samples)


def get_generator_for_samples_in_csv(path: str, preprocess: Callable = None) -> Iterator[np.array]:
    """
    get generator[np.ndarray] for samples in csv. notice that the first row (titles) are being ignored.
    :param path: string that contains the path for the csv.
    :param preprocess: (optional) function for preprocess the data. the function can change the values by reference,
                        can change by value (the function should return specific sample), or can remove sample due to a
                        condition (the function should return []).
    :return: generator[np.ndarray] for the samples.
    """
    data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")
    for row in data_frame.values:
        sample = list(row)
        if preprocess is not None:
            sample = preprocess(sample)
        if sample:
            yield np.array(sample)


def print_graph(x_values: list, y_values: list, x_label: str, y_label: str):
    """
    print graph according to the given parameters.
    :param x_values: list the contains the values in axis 'x'.
    :param y_values: list the contains the values in axis 'y'.
    :param x_label: title for axis 'x'.
    :param y_label: title for axis 'y'.
    """
    plt.plot(x_values, y_values, 'ro')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# deprecated
def complete_features(samples: Sample, given_features: list[int], total_features_num: int,
                      default_value: float = np.inf) -> Sample:
    """
    expands each of the given samples to size total_features_num by placing default_value in all the places which are
    not in given_features.
    samples has to be arranged according to given_features- the function sets the first value in sample to the first
    index in given_features in the new expanded sample.
    given_features isn't required to be sorted.
    total_features_num has to be equal or above than the size of sample and given_features.
    :param samples: given samples for expanding.
    :param given_features: list of the indices of the chosen features.
    :param total_features_num: the number to be expanded to.
    :param default_value: the default value to place in all the places which are not in given_features. default value is inf.
    :return expanded sample of shape (n_samples, total_features_num).
    """
    expanded_samples = []
    for sample in samples:
        expanded_sample = [default_value for _ in range(total_features_num)]
        for feature_idx, feature_value in zip(given_features, sample):
            expanded_sample[feature_idx] = feature_value
        expanded_samples.append(expanded_sample)
    return np.array(expanded_samples)


def normalize_data(data):
    return pre.normalize(data, axis=0)


""""""""""""""""""""""""""""""""""""""""""" Score Function """""""""""""""""""""""""""""""""""""""""""


def get_correlation_to_feature(dataset, target, feature):
    """
    get dataset, feature and target feature, and check the feature correlation to the target feature.
    need to get normalized dataset !
    :param dataset: the dataset we are working on.
    :param target: the target feature index.
    :param feature : the index of the feature we want to check how correlated it to target feature.
    :return: the correlation between two features.
    """
    return abs(np.correlate(dataset[:, target], dataset[:, feature])[0])


def get_correlation_to_other_features(dataset, features, feature):
    """
    get dataset, feature and set of given features, and check the feature correlation to the whole set.
    need to get normalized dataset !
    :param dataset: the dataset we are working on.
    :param features: set of given features(indexes).
    :param feature : the index of the feature we want to check how correlated it to set of given features.
    :return: the correlation between two features.
    """
    return np.mean([get_correlation_to_feature(dataset, f, feature) for f in features])


def get_price_score(feature, costs_list):
    """
    get feature index and list of costs, and return the feature cost.
    :param feature: the index of the wanted feature.
    :param costs_list: list that include all the feature's costs.
    :return: the given feature cost.
    """
    return costs_list[feature]


def get_certainty(dataset, target, features, feature, learning_algo):
    """
    return the level of the certainty according to the theory we explain in the PDF.
    :param dataset: the dataset we are working on.
    :param target: the target feature index.
    :param features: set of given features(indexes).
    :param feature: the index of the wanted feature.
    :param learning_algo: untrained classifier.
    :return: the level of the certainty.
    """
    new_features = np.append(features, [feature])
    data = pd.DataFrame(dataset)
    learning_algo.fit(data[new_features], data[target])
    probabilities = learning_algo.predict_proba(data[new_features])
    certainty = stats.entropy(probabilities[0], probabilities[1])
    return 1 - certainty


def score_function_a(dataset, features, feature, target, costs_list, alpha=1):
    """
    return the feature score according to the theory we explain in the PDF.
    """
    return (get_correlation_to_feature(dataset, target, feature) +
            alpha * get_correlation_to_other_features(dataset, features, feature)) / get_price_score(feature,
                                                                                                     costs_list)


def score_function_b(dataset, features, feature, target, costs_list, learning_algo=None):
    """
    return the feature score according to the theory we explain in the PDF.
    """
    return get_certainty(dataset, target, features, feature, learning_algo) / get_price_score(feature,
                                                                                              costs_list)