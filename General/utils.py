"""
Utils For The Project
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.preprocessing as pre

from typing import Callable, Tuple, Union, Optional
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


def get_samples_from_csv(path: str, class_index: int = 0, preprocess: Optional[Callable] = None, **kw) -> Tuple[np.array, Classes]:
    """
    get samples and classes from csv as np.array. notice that the first row (titles) are being ignored.
    :param path: string that contains the path for the csv.
    :param class_index: the index of the class in the sample. for default, the class will be at the first place.
    :param preprocess: (optional) function for preprocess the data. the function runs over the rows and may change the
        data by reference, by value (the function should return specific sample), or can remove sample due to a condition
        (in such case the function should return []).
    :return: tuple of samples and class, both are as np.array.
    """
    dataset = pd.read_csv(filepath_or_buffer=path, sep=",").to_numpy()
    if preprocess is not None:
        np.apply_along_axis(preprocess, 1, dataset, **kw)
    complementary_list = list(get_complementary_set(range(dataset.shape[1]), [class_index]))
    return dataset[:, complementary_list], dataset[:, [class_index]].flatten()


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


def is_number(value: str) -> bool:
    """
    Returns if value is number or not. number is considered whole number or fraction. exponent is not considered as number.
    :param value: string.
    :return: True if value is number, False otherwise.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_dataset(path: str, class_index: int = 0, train_ratio=0.25, random_seed: Optional[int] = None,
                shuffle: Optional[bool] = True, **kw) -> Tuple[TrainSamples, TestSamples]:
    """
    Gets dataset from csv. the function reads csv from given path, processes it, and returns it as Tuple of TrainSamples, TestSamples.
    :param path: string that contains the path for the csv.
    :param class_index: the index of the class in the sample. for default, the class will be at the first place.
    :param train_ratio: if float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include
        in the train split. if int, represents the absolute number of train samples. if None, the value is automatically
        set to the complement of the test size.
    :param random_seed: controls the shuffling applied to the data before applying the split. Pass an int for reproducible
        output across multiple function calls.
    :param shuffle: whether or not to shuffle the data before splitting.
    :return: Tuple of TrainSamples, TestSamples.
    """
    categories = {}
    samples, classes = get_samples_from_csv(path=path,
                                            class_index=class_index,
                                            preprocess=categorical_to_numeric,
                                            categories=categories,
                                            **kw)
    shuffle = shuffle if type(random_seed) == int else False
    train_samples, test_samples, train_classes, test_classes = sklearn.model_selection.train_test_split(samples,
                                                                                                        classes,
                                                                                                        test_size=train_ratio,
                                                                                                        random_state=random_seed,
                                                                                                        shuffle=shuffle,
                                                                                                        stratify=None if not shuffle else True)
    return TrainSamples(train_samples, train_classes), TestSamples(test_samples, test_classes)


def get_complementary_set(elements_list: Union, existing_elements: Union) -> set:
    """
    Gets the complementary set of given set, i.e set of the elements in elements_list which are not in existing_elements.
    :param elements_list: set of the all elements.
    :param existing_elements: set of some elements that shouldn't be in the complementary set.
    :return: set of the elements in elements_list which are not in existing_elements.
    """
    return set(elements_list) - set(existing_elements)


def normalize_data(data):
    """
    Normalizes given data.
    :param data: data for normalizing.
    :return: normalized data.
    """
    return pre.normalize(data, axis=0)


# deprecated
def complete_features(samples: Sample, given_features: list[int], total_features_num: int,
                      default_value: Optional[float] = np.inf) -> Sample:
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


