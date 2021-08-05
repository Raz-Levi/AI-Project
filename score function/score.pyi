"""
Score function for the algorithm
"""

""""""""""""""""""""""""""""""""""""""""""" Imports """""""""""""""""""""""""""""""""""""""""""
from numpy import mean

def get_correlation_to_feature(dataset, target, feature):
    return abs(dataset[target].corr()[feature])

def get_correlation_to_other_features(dataset, features, feature):
    return mean([get_correlation_to_feature(dataset, f, feature) for f in features])

def get_price_score(feature, costs_dict):
    return (sorted(costs_dict, reverse=True,key=costs_dict.get)).index(feature)

