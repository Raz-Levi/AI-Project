"""
This module contains naive algorithms
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from General.utils import *
from LearningAlgorithms.abstract_algorithm import LearningAlgorithm, SequenceAlgorithm

""""""""""""""""""""""""""""""""""""""""""" Classes """""""""""""""""""""""""""""""""""""""""""


class EmptyAlgorithm(LearningAlgorithm):
    """
    A naive algorithm that ignores the option to buy features and runs sklearn's classifier for samples filtering the\
    given features only.
    """
    def __init__(self, classifier: sklearn.base.ClassifierMixin):
        """
        Init function.
        :param classifier: sklearn's classifier. the function saves it and uses it later.
        """
        super().__init__()
        self._train_samples = None
        self._classifier = classifier

    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        """
        Trains the classifier. the function saves the train samples for the prediction.
        :param train_samples: training dataset contains training data of shape (n_samples, n_features), i.e samples are
        in the rows, and target values of shape (n_samples,). the function saves it.
        :param features_costs: list in length number of features that contains the costs of each feature according to
        indices. in first index you will find the cost of the first feature, etc.
        """
        self._train_samples = train_samples

    def predict(self, samples: TestSamples, given_features: GivenFeatures, maximal_cost: float) -> Classes:
        """
        Predicts the class labels for the provided data. the function runs the fit and predict function of the given
        sklearn's classifier on the given test samples filtering the given features only.
        :param samples: test samples of shape (n_samples, n_features), i.e samples are in the rows.
        :param given_features: list of the indices of the chosen features.
        :param maximal_cost: the maximum available cost for buying features.
        :return: Classes of shape (n_samples,) contains the class labels for each data sample.
        """
        self._classifier.fit(self._train_samples.samples[:, given_features], self._train_samples.classes)
        return self._classifier.predict(samples[:, given_features])


class RandomAlgorithm(SequenceAlgorithm):
    """
    A naive algorithm that chooses randomly features to add to the given features according to the given budget.
    """
    def __init__(self, classifier: sklearn.base.ClassifierMixin):
        super().__init__(classifier)

    # Private Methods
    def _buy_features(self, given_features: GivenFeatures, maximal_cost: float) -> GivenFeatures:
        """
        A method for choosing the supplementary features. the methods chooses features randomly and stops if all the
        features were chosen or if the cost of the supplementary features is above the maximal cost.
        :param given_features: list of the indices of the chosen features.
        :param maximal_cost: the maximum available cost for buying features.
        :return: the updated given features including all the chosen features.
        """
        available_features = get_complementary_set(range(self._train_samples.get_features_num()), given_features)
        while len(available_features) and maximal_cost:
            chosen_feature = random.choice(list(available_features))
            available_features -= {chosen_feature}
            if self._features_costs[chosen_feature] <= maximal_cost:
                maximal_cost -= self._features_costs[chosen_feature]
                given_features.append(chosen_feature)
        return given_features


class OptimalAlgorithm(SequenceAlgorithm):
    """
    A naive algorithm that chooses the cheapest features to add to the given features according to the given budget.
    """
    def __init__(self, classifier: sklearn.base.ClassifierMixin):
        super().__init__(classifier)

    # Private Methods
    def _buy_features(self, given_features: GivenFeatures, maximal_cost: float) -> GivenFeatures:
        """
        A method for choosing the supplementary features. the methods chooses features according to their costs. the
        features are sorted according to their costs, for the cheapest to the most expensive. the features are chosen
        from the cheapest until all the features are chosen or the costs of the chosen features is above the maximal cost.
        features were chosen or if the cost of the supplementary features is above the maximal cost.
        :param given_features: list of the indices of the chosen features.
        :param maximal_cost: the maximum available cost for buying features.
        :return: the updated given features including all the chosen features.
        """
        available_features = list(get_complementary_set(range(self._train_samples.get_features_num()), given_features))
        available_features.sort(reverse=True, key=lambda a: self._features_costs[a])
        while len(available_features) and maximal_cost:
            chosen_feature = available_features.pop()
            if self._features_costs[chosen_feature] > maximal_cost:
                break
            maximal_cost -= self._features_costs[chosen_feature]
            given_features.append(chosen_feature)
        return given_features
