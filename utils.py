"""
Utils For The Project
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, Iterator

""""""""""""""""""""""""""""""""""" Definitions and Consts """""""""""""""""""""""""""""""""""


""""""""""""""""""""""""""""""""""""""""""" Methods """""""""""""""""""""""""""""""""""""""""""


def get_samples_from_csv(path: str, preprocess: Callable = None) -> np.ndarray:
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


def get_generator_for_samples_in_csv(path: str, preprocess: Callable = None) -> Iterator[np.ndarray]:
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
