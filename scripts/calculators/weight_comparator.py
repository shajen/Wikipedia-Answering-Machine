from collections import defaultdict
from data.models import *
from scipy.spatial.distance import cdist
import logging
import math
import cupy as cp
import operator
import re
# import tensorflow as tf

class WeightComparator:
    def __init__(self, method_name, sum_neighbors, ascending_order, distance_function):
        self.__method_name = method_name
        self.__sum_neighbors = sum_neighbors
        self.__ascending_order = ascending_order
        self.__distance_function = distance_function

    def method(self):
        return self.__method_name + (", n: %03d, m: %s" % (self.__sum_neighbors, self.__distance_function))

    def ascending_order(self):
        return self.__ascending_order

    def get_best_score(self, question_vector, vectors, words_set_weights):
        distances = cdist([question_vector], vectors, self.__distance_function)
        if self.__ascending_order:
            i = cp.argmax(distances[0])
        else:
            i = cp.argmin(distances[0])
        return (distances[0][i], i)

class CosineWeightComparator(WeightComparator):
    def __init__(self, method_name, sum_neighbors):
        super().__init__(method_name, sum_neighbors, False, 'cosine')

class EuclideanWeightComparator(WeightComparator):
    def __init__(self, method_name, sum_neighbors):
        super().__init__(method_name, sum_neighbors, False, 'euclidean')

class CityblockWeightComparator(WeightComparator):
    def __init__(self, method_name, sum_neighbors):
        super().__init__(method_name, sum_neighbors, False, 'cityblock')

class TfIdfWeightComparator(WeightComparator):
    def __init__(self, method_name, sum_neighbors, power_factor):
        self.__method_name = method_name
        self.__sum_neighbors = sum_neighbors
        self.__power_factor = power_factor

    def method(self):
        return self.__method_name + (", n: %03d, pf: %.2f" % (self.__sum_neighbors, self.__power_factor))

    def ascending_order(self):
        return True

    def get_best_score(self, question_vector, vectors, words_set_weights):
        # distances = cp.power(cp.sum(vectors, axis=1), cp.count_nonzero(vectors, axis=1))
        # distances = tf.math.pow(tf.math.reduce_sum(tf.constant(vectors, dtype=tf.float32), axis=1), tf.dtypes.cast(tf.math.count_nonzero(vectors, axis=1), tf.float32)).cupy()
        distances = []
        for weights in words_set_weights:
            distances.append(sum(weights.values()) * math.pow(len(weights), self.__power_factor))
        i = cp.argmax(distances)
        return (distances[i], i)
