"""
Utils For The Project
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

import abc
from typing import Callable, Iterator, Tuple
from dataclasses import dataclass

""""""""""""""""""""""""""""""""""" Definitions and Consts """""""""""""""""""""""""""""""""""

Sample = np.array
Classes = np.array


@dataclass
class TrainSamples:
    samples: np.array  # Training data- shape (n_samples, n_features), i.e samples are in the rows.
    classes: Classes  # Target values- shape (n_samples,).


@dataclass
class TestSamples:
    samples: np.array  # Test data- shape (n_samples, n_features), i.e samples are in the rows.
    classes: Classes  # Target values- shape (n_samples,) for computing the accuracy.


""""""""""""""""""""""""""""""""""""""""""" Methods """""""""""""""""""""""""""""""""""""""""""


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


def get_samples_from_csv(path: str, preprocess: Callable = None, include_first_column: bool = True, **kw) -> np.array:
    """
    get samples from csv as np.ndarray. notice that the first row (titles) are being ignored.
    :param path: string that contains the path for the csv.
    :param preprocess: (optional) function for preprocess the data. the function can change the values by reference,
                        can change by value (the function should return specific sample), or can remove sample due to a
                        condition (the function should return []).
    :param include_first_column: if True, the first column of the dataset is included. if False, the first column of the
    dataset is not included.
    :return: samples as np.ndarray.
    """
    samples, data_frame = [], pd.read_csv(filepath_or_buffer=path, sep=",")
    for example in data_frame.values:
        sample = list(example) if include_first_column else list(example)[1:]
        if preprocess is not None:
            processed_sample = preprocess(sample, **kw)
            sample = sample if processed_sample is None else processed_sample
        if sample:
            samples.append(sample)
    return np.array(samples)


def get_generator_for_samples_in_csv(path: str, preprocess: Callable = None, include_first_column: bool = True, **kw) -> Iterator[np.array]:
    """
    get generator[np.ndarray] for samples in csv. notice that the first row (titles) are being ignored.
    :param path: string that contains the path for the csv.
    :param preprocess: (optional) function for preprocess the data. the function can change the values by reference,
                        can change by value (the function should return specific sample), or can remove sample due to a
                        condition (the function should return []).
    :param include_first_column: if True, the first column of the dataset is included. if False, the first column of the
    dataset is not included.
    :return: generator[np.ndarray] for the samples.
    """
    data_frame = pd.read_csv(filepath_or_buffer=path, sep=",")
    for row in data_frame.values:
        sample = list(row) if include_first_column else list(row)[1:]
        if preprocess is not None:
            sample = preprocess(sample, **kw)
        if sample:
            yield np.array(sample)


def categorical_to_numeric(sample: Sample, categories: dict):
    """
    Preprocess for samples- the alphabetic features will be converted to numerical features.
    :param sample: the sample for converting.
    :param categories: a dictionary that contains the converted features and their numerical values.
    """
    for feature_num in range(len(sample)):
        if not is_number(sample[feature_num]):
            if sample[feature_num] not in categories.keys():
                if f'_categories{feature_num}' not in categories.keys():
                    categories[f'_categories{feature_num}'] = 0
                categories[sample[feature_num]] = categories[f'_categories{feature_num}']
                sample[feature_num] = categories[f'_categories{feature_num}']
                categories[f'_categories{feature_num}'] += 1
            else:
                sample[feature_num] = categories[sample[feature_num]]


def get_dataset(path: str, train_ratio: float = 0.7, preprocess: Callable = None) -> Tuple[TrainSamples, TestSamples]:
    pass


def is_number(value: str) -> bool:
    """
    Returns if value is number or not. number is considered whole number or fraction. exponent is not considered as number.
    :param value: string.
    :return: True if value is number, False otherwise.
    """
    return type(value) == int or type(value) == float or (type(value) == str and value.replace('.', '', 1).isnumeric())


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
