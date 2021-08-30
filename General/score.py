""""
this module defines the score functions we are using
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from General.utils import *
import scipy.stats as stats

""""""""""""""""""""""""""""""""""""""""""" Class """""""""""""""""""""""""""""""""""""""""""""


class ScoreFunction(abc.ABC):
    """
    An abstract class for ScoreFunction.
    """
    # Public Methods
    def __init__(self, classifier: sklearn.base.ClassifierMixin = None, alpha: int = 1):
        super().__init__()
        self._classifier = classifier
        self._alpha = alpha

    def __call__(self, *args, **kwargs):
        return self._execute_function(*args, **kwargs)

    # Private Methods
    @abc.abstractmethod
    def _execute_function(self, train_samples: TrainSamples, given_features: GivenFeatures,
                          new_feature: int, costs_list: list[float]) -> float:
        """"
        this function will execute the score function
        """
        ...


class ScoreFunctionA(ScoreFunction):
    """"
    return the feature score according to the theory we explain in the PDF.
    """
    # Public Methods
    def __init__(self, alpha: int = 1, classifier: sklearn.base.ClassifierMixin = None):
        super().__init__(classifier, alpha)

    # Private Methods
    def _execute_function(self, train_samples: TrainSamples, given_features: GivenFeatures,
                          new_feature: int, costs_list: list[float]) -> float:
        price = costs_list[new_feature]
        frac = (self._get_correlation_to_feature(train_samples.classes, new_feature, train_samples) /
                self._alpha * self._get_correlation_to_given_features(train_samples, new_feature, given_features))
        return frac / price

    def _get_correlation_to_given_features(self, train_samples, new_feature, given_features):
        return np.mean([self._get_correlation_to_feature(train_samples.samples[int(f)], new_feature, train_samples)
                        for f in given_features])

    @staticmethod
    def _get_correlation_to_feature(feature1, feature2, train_samples):
        return np.abs(np.correlate(feature1, train_samples.samples[feature2])[0])


class ScoreFunctionB(ScoreFunction):
    """"
    return the feature score according to the theory we explain in the PDF.
    """
    # Public Methods
    def __init__(self, alpha: int = 1, classifier: sklearn.base.ClassifierMixin = None):
        super().__init__(classifier, alpha)

    # Private Methods
    def _execute_function(self, train_samples: TrainSamples, given_features: GivenFeatures,
                          new_feature: int, costs_list: list[float]) -> float:
        price = costs_list[new_feature]
        certainty = self._get_certainty(train_samples, given_features, new_feature)
        return certainty / price

    def _get_certainty(self, train_samples: TrainSamples, given_features: GivenFeatures,
                       new_feature: int):
        """
        return the level of certainty according to the theory we explain in the PDF.
        :return: level of certainty.
        """
        new_features = np.append(given_features, [new_feature])
        data = pd.DataFrame(train_samples.samples)
        self._learning_algorithm.fit(data[new_features], train_samples.classes)
        probabilities = self._learning_algorithm.predict_proba(data[new_features])
        certainty = self._calc_total_certainty(probabilities)
        return 1 - certainty

    @staticmethod
    def _calc_total_certainty(probabilities):
        res = 0
        for row in probabilities:
            res += stats.entropy(row)
        return res / len(probabilities)
