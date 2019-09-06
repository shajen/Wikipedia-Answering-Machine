from collections import defaultdict
from data.models import *
from scipy.spatial.distance import cdist
import calculators.weight_calculator
import logging
import math
import numpy as np
import re

class VectorWeightCalculator(calculators.weight_calculator.WeightCalculator):
    def __init__(self, debug_top_items):
        super().__init__(debug_top_items)

    def __dict_to_vector(self, keys, d):
        v = []
        for key in keys:
            try:
                v.append(d[key])
            except:
                v.append(0.0)
        return v

    def _vector_upload_positions(self, question, method_name, sum_neighbors, ascending_order, distance_function, question_words_weight, articles_words_weight):
        logging.info('')
        logging.info(distance_function + (' vector %d neighbors' % sum_neighbors))

        articles_weight = {}
        keys = question_words_weight.keys()
        question_vector = self.__dict_to_vector(keys, question_words_weight)
        articles_vectors = []
        for article in articles_words_weight:
            articles_vectors.append(self.__dict_to_vector(keys, articles_words_weight[article]))
        logging.debug('finished vectors preparing')

        distances = cdist([question_vector], articles_vectors, distance_function)
        i = 0
        for article in articles_words_weight:
            articles_weight[article] = distances[0][i]
            i += 1
        logging.info('finished weights calculations')

        positions = self._count_positions(question, articles_words_weight, articles_weight, ascending_order, Article.objects, Word.objects)
        self._upload_positions(positions, method_name + (", type: %s_vector_%03d_neighbors" % (distance_function, sum_neighbors)))
        return positions

class CosineVectorWeightCalculator(VectorWeightCalculator):
    def __init__(self, debug_top_items):
        super().__init__(debug_top_items)

    def upload_positions(self, question, method_name, sum_neighbors, question_words_weight, articles_words_weight):
        return self._vector_upload_positions(question, method_name, sum_neighbors, False, 'cosine', question_words_weight, articles_words_weight)

class EuclideanVectorWeightCalculator(VectorWeightCalculator):
    def __init__(self, debug_top_items):
        super().__init__(debug_top_items)

    def upload_positions(self, question, method_name, sum_neighbors, question_words_weight, articles_words_weight):
        return self._vector_upload_positions(question, method_name, sum_neighbors, False, 'euclidean', question_words_weight, articles_words_weight)

class CityblockVectorWeightCalculator(VectorWeightCalculator):
    def __init__(self, debug_top_items):
        super().__init__(debug_top_items)

    def upload_positions(self, question, method_name, sum_neighbors, question_words_weight, articles_words_weight):
        return self._vector_upload_positions(question, method_name, sum_neighbors, False, 'cityblock', question_words_weight, articles_words_weight)
