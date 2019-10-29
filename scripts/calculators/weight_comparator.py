from collections import defaultdict
from data.models import *
from scipy.spatial.distance import cdist
import calculators.weight_calculator
import logging
import math
import numpy as np
import operator
import re

class WeightComparator:
    def __init__(self, method_name, sum_neighbors, ascending_order, distance_function):
        self.__method_name = method_name
        self.__sum_neighbors = sum_neighbors
        self.__ascending_order = ascending_order
        self.__distance_function = distance_function

    def method(self):
        return self.__method_name + (", type: %s_vector_%03d_neighbors" % (self.__distance_function, self.__sum_neighbors))

    def ascending_order(self):
        return self.__ascending_order

    def __dict_to_vector(self, keys, d):
        v = []
        for key in keys:
            try:
                v.append(d[key])
            except:
                v.append(0.0)
        return v

    def get_best_score(self, question_words_weights, words_set_weights):
        articles_weight = {}
        keys = question_words_weights.keys()
        # logging.info(question_words_weights)
        # keys = list(map(lambda x: x[0], sorted(question_words_weights.items(), key=operator.itemgetter(1), reverse=True)))
        # logging.info(keys)
        # keys = keys[:3]
        question_vector = self.__dict_to_vector(keys, question_words_weights)
        vectors = []
        for weights in words_set_weights:
            vectors.append(self.__dict_to_vector(keys, weights))

        distances = cdist([question_vector], vectors, self.__distance_function)
        if self.__ascending_order:
            i = np.argmax(distances[0])
        else:
            i = np.argmin(distances[0])
        return (distances[0][i], words_set_weights[i])

class CosineWeightComparator(WeightComparator):
    def __init__(self, method_name, sum_neighbors):
        super().__init__(method_name, sum_neighbors, False, 'cosine')

class EuclideanWeightComparator(WeightComparator):
    def __init__(self, method_name, sum_neighbors):
        super().__init__(method_name, sum_neighbors, False, 'euclidean')

class CityblockWeightComparator(WeightComparator):
    def __init__(self, method_name, sum_neighbors):
        super().__init__(method_name, sum_neighbors, False, 'cityblock')

class TfIdfWeightComparator:
    def __init__(self, method_name, sum_neighbors, power_factor):
        self.__method_name = method_name
        self.__sum_neighbors = sum_neighbors
        self.__power_factor = power_factor

    def method(self):
        return self.__method_name + (", pf: %.2f, type: tf_idf_%03d_neighbors" % (self.__power_factor, self.__sum_neighbors))

    def ascending_order(self):
        return True

    def get_best_score(self, question_words_weights, words_set_weights):
        distances = []
        for weights in words_set_weights:
            distances.append(sum(weights.values()) * math.pow(len(weights), self.__power_factor))
        i = np.argmax(distances)
        return (distances[i], words_set_weights[i])