from collections import defaultdict
from data.models import *
from scipy.spatial import distance
import calculators.weight_calculator
import logging
import math
import numpy as np
import re

class VectorWeightCalculator(calculators.weight_calculator.WeightCalculator):
    def __init__(self, debug_top_items):
        super().__init__(debug_top_items)

    def __dict_distance(self, keys, distance_function, d1, d2):
        v1 = []
        v2 = []
        for key in keys:
            try:
                v1.append(d1[key])
            except:
                v1.append(0.0)
            try:
                v2.append(d2[key])
            except:
                v2.append(0.0)
        return distance_function(v1, v2)

    def _vector_upload_positions(self, question, method_name, sum_neighbors, ascending_order, distance_function, question_words_weight, articles_words_weight):
        logging.info('')
        if sum_neighbors:
            logging.info(distance_function.__name__ + ' vector neighbors')
        else:
            logging.info(distance_function.__name__ + ' vector')

        articles_weight = {}
        keys = question_words_weight.keys()
        for article in articles_words_weight:
            articles_weight[article] = self.__dict_distance(keys, distance_function, question_words_weight, articles_words_weight[article])

        positions = self._count_positions(question, articles_words_weight, articles_weight, ascending_order, Article.objects, Word.objects)
        if sum_neighbors:
            self._upload_positions(positions, method_name + (", type: %s_vector_neighbours" % distance_function.__name__))
        else:
            self._upload_positions(positions, method_name + (", type: %s_vector" % distance_function.__name__))
        return positions

class CosineVectorWeightCalculator(VectorWeightCalculator):
    def __init__(self, debug_top_items):
        super().__init__(debug_top_items)

    def upload_positions(self, question, method_name, sum_neighbors, question_words_weight, articles_words_weight):
        return self._vector_upload_positions(question, method_name, sum_neighbors, False, distance.cosine, question_words_weight, articles_words_weight)

class EuclideanVectorWeightCalculator(VectorWeightCalculator):
    def __init__(self, debug_top_items):
        super().__init__(debug_top_items)

    def upload_positions(self, question, method_name, sum_neighbors, question_words_weight, articles_words_weight):
        return self._vector_upload_positions(question, method_name, sum_neighbors, False, distance.euclidean, question_words_weight, articles_words_weight)

class CityblockVectorWeightCalculator(VectorWeightCalculator):
    def __init__(self, debug_top_items):
        super().__init__(debug_top_items)

    def upload_positions(self, question, method_name, sum_neighbors, question_words_weight, articles_words_weight):
        return self._vector_upload_positions(question, method_name, sum_neighbors, False, distance.cityblock, question_words_weight, articles_words_weight)
