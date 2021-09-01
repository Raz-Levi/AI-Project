""""
This module contains a search algorithm based on a genetic algorithm
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from General.utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from deap import creator, base, tools, algorithms
from sklearn.model_selection import train_test_split

from LearningAlgorithms.abstract_algorithm import SequenceAlgorithm
""""""""""""""""""""""""""""""""""""""""""" Classes """""""""""""""""""""""""""""""""""""""""""


class GeneticAlgorithm(SequenceAlgorithm):
    """
    A search algorithm based on a genetic algorithm.
    """

    def __init__(self, classifier: sklearn.base.ClassifierMixin):
        super().__init__(classifier)
        self._all_features = None
        self._max_cost = None
        self._given_features = None

    def fit(self, train_samples: TrainSamples, features_costs: list[float]):
        """
        Trains the classifier. the function saves the train samples and the features costs for the prediction.
        :param train_samples: training dataset contains training data of shape (n_samples, n_features), i.e samples are
        in the rows, and target values of shape (n_samples,). the function saves it.
        :param features_costs: list in length number of features that contains the costs of each feature according to
        indices. in first index you will find the cost of the first feature, etc. the function saves it.
        """
        self._train_samples = train_samples
        self._features_costs = features_costs
        self._all_features = [i for i in range(train_samples.get_features_num())]

    def _buy_features(self, given_features: GivenFeatures, maximal_cost: float) -> GivenFeatures:
        """
        this function choose from the best subsets of features calculated by the genetic algorithm
        the one which is valid according to our parameters
        :param given_features: the features that are given to us for free
        :param maximal_cost: the limit of all the bought features cost
        :return: a valid subset of the features
        """
        self._max_cost = maximal_cost
        self._given_features = given_features

        max_val_subsets = self._get_max_val_subsets()
        valid = self._get_valid_subset(max_val_subsets)
        return valid

    def _get_max_val_subsets(self) -> List[List[int]]:
        """
        the genetic algorithm. return the HallOfFame - the best feature's subsets it terms of accuracy
        :return: HOF
        """
        X_trainAndTest, X_validation, y_trainAndTest, y_validation = train_test_split(self._train_samples.samples,
                                                                                      self._train_samples.classes,
                                                                                      test_size=0.20, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_trainAndTest, y_trainAndTest, test_size=0.20,
                                                            random_state=42)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(self._all_features))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self._get_fitness, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        hof = self._get_HOF(toolbox)
        testAccuracyList, validationAccuracyList, individualList, percentileList = self._get_metrics(hof,
                                                                                                     X_trainAndTest,
                                                                                                     X_validation,
                                                                                                     y_trainAndTest,
                                                                                                     y_validation)

        # Get a list of subsets that performed best on validation data
        maxValAccSubsetIndices = [index for index in range(len(validationAccuracyList)) if
                                  validationAccuracyList[index] == max(validationAccuracyList)]
        maxValIndividuals = [individualList[index] for index in maxValAccSubsetIndices]
        maxValSubsets = [[self._all_features[index] for index in range(len(individual)) if individual[index] == 1] for
                         individual in maxValIndividuals]

        return maxValSubsets

    def _get_valid_subset(self, subsets: List[List[int]]) -> List[int]:
        """
        this function return the first item in the valid subsets list if exist.
        :param subsets: subset of features.
        :return: a valid subset of features.
        """
        valid = []
        for subset in subsets:
            if self._is_legal_subset(subset):
                valid.append(subset)
        if not len(valid):
            raise ValueError("No Valid Solution")
        return valid[0]

    def _calc_subset_cost(self, subset: list[int]) -> float:
        """
        this function calculate the cost of a given subset
        :param subset: subset of features.
        :return: cost.
        """
        total_cost = 0
        for item in subset:
            total_cost += self._features_costs[item]
        return total_cost

    def _is_legal_subset(self, individual) -> bool:
        """
        this function determine rather a subset id legal in terms of cost.
        :param individual: a "genome" - subset of features.
        :return: the legality of the given genome.
        """
        added_features = get_complementary_set(individual, self._given_features)
        cost = self._calc_subset_cost(list(added_features))
        if cost > self._max_cost:
            return False
        return True

    def _get_fitness(self, individual: List[int], X_train, X_test, y_train, y_test) -> (float,):
        """
        the fitness function for the genetic algorithm.
        :param individual: genome.
        :return: accuracy
        """

        # accuracy is between 0 to 1, scoring a subset with -1 when the initial population is legal
        # promise that this subset won't chosen

        if not self._is_legal_subset(individual):
            return -1

        x_train_data = pd.DataFrame(X_train)
        x_test_data = pd.DataFrame(X_test)

        unwanted_cols = [index for index in range(len(individual)) if individual[index] == 0]
        X_trainParsed = x_train_data.drop(x_train_data.columns[unwanted_cols], axis=1)
        X_trainOhFeatures = pd.get_dummies(X_trainParsed)
        X_testParsed = x_test_data.drop(x_test_data.columns[unwanted_cols], axis=1)
        X_testOhFeatures = pd.get_dummies(X_testParsed)

        sharedFeatures = set(X_trainOhFeatures.columns) & set(X_testOhFeatures.columns)
        removeFromTrain = set(X_trainOhFeatures.columns) - sharedFeatures
        removeFromTest = set(X_testOhFeatures.columns) - sharedFeatures
        X_trainOhFeatures = X_trainOhFeatures.drop(list(removeFromTrain), axis=1)
        X_testOhFeatures = X_testOhFeatures.drop(list(removeFromTest), axis=1)

        clf = LogisticRegression(max_iter=200)
        clf.fit(X_trainOhFeatures, y_train)
        predictions = clf.predict(X_testOhFeatures)
        accuracy = accuracy_score(y_test, predictions)

        # Return calculated accuracy as fitness
        return accuracy,

    @staticmethod
    def _get_HOF(toolbox):
        """
        return the hall of fame - the best individuals (Genome : List[int]) in the population (List[List[int]]).
        :param toolbox:
        :return: HOF
        """
        numPop = 50
        numGen = 8
        pop = toolbox.population(numPop * numGen)
        hof = tools.HallOfFame(numPop * numGen)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Launch genetic algorithm
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=numGen, stats=stats, halloffame=hof,
                                       verbose=True)

        # Return the hall of fame
        return hof

    def _get_metrics(self, hof, X_trainAndTest, X_validation, y_trainAndTest, y_validation):
        # Get list of percentiles in the hall of fame
        percentileList = [i / (len(hof) - 1) for i in range(len(hof))]

        # Gather fitness data from each percentile
        testAccuracyList = []
        validationAccuracyList = []
        individualList = []
        for individual in hof:
            testAccuracy = individual.fitness.values
            validationAccuracy = self._get_fitness(individual, X_trainAndTest, X_validation, y_trainAndTest,
                                                   y_validation)
            testAccuracyList.append(testAccuracy[0])
            validationAccuracyList.append(validationAccuracy[0])
            individualList.append(individual)
        testAccuracyList.reverse()
        validationAccuracyList.reverse()
        return testAccuracyList, validationAccuracyList, individualList, percentileList
