from collections import defaultdict
from data.models import *
import calculators.weight_calculator
import logging
import math
import numpy as np
import re

class TfIdfWeightCalculator(calculators.weight_calculator.WeightCalculator):
    def __init__(self, debug_top_items):
        super().__init__(debug_top_items)

    def __count_tf_idf(self, question, is_title, sum_neighbors):
        words = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', question.name)
        words = Word.objects.filter(changed_form__in=words, is_stop_word=False).values('id')
        words = list(map(lambda x: x['id'], words))

        articles_words_count = defaultdict(lambda: defaultdict(lambda: 0))
        articles_words_positions = defaultdict(defaultdict)
        articles_positions = defaultdict(list)
        if sum_neighbors:
            occurrences = Occurrence.objects.filter(word_id__in=words, is_title=is_title).values('article_id', 'word_id', 'positions_count', 'positions')
        else:
            occurrences = Occurrence.objects.filter(word_id__in=words, is_title=is_title).values('article_id', 'word_id', 'positions_count')
        for occurrence in occurrences:
            articles_words_count[occurrence['article_id']][occurrence['word_id']] += occurrence['positions_count']
            if sum_neighbors: #and len(positions) == occurrence['positions_count']:
                positions = [int(p) for p in occurrence['positions'].strip().split(',') if p]
                articles_words_positions[occurrence['article_id']][occurrence['word_id']] = positions
                articles_positions[occurrence['article_id']].extend([(p, occurrence['word_id']) for p in positions])

        words_articles_count = defaultdict(lambda: 0)
        for item_id in articles_words_count:
            for word_id in articles_words_count[item_id]:
                words_articles_count[word_id] += 1

        words_idf = {}
        for word_id in words_articles_count:
            words_idf[word_id] = math.log(len(self.articles_title_count) / words_articles_count[word_id])

        question_words_weights = {}
        for word in words:
            try:
                question_words_weights[word] = words_idf[word] / len(words)
            except:
                pass

        articles_words_weights = defaultdict(defaultdict)
        for item_id in articles_words_count:
            for word_id in articles_words_count[item_id]:
                if sum_neighbors:
                    w = 0.0
                    for position in articles_words_positions[item_id][word_id]:
                        positions = [word_id for (p, word_id) in articles_positions[item_id] if abs(p - position) <= 2]
                        positions = np.unique(positions)
                        w += math.pow(1.7, len(positions)) / articles_words_count[item_id][word_id]
                    articles_words_weights[item_id][word_id] = w * words_idf[word_id]
                else:
                    if is_title:
                        tf = articles_words_count[item_id][word_id] / self.articles_title_count[item_id]
                    else:
                        tf = articles_words_count[item_id][word_id] / self.articles_content_count[item_id]
                    articles_words_weights[item_id][word_id] = tf * words_idf[word_id]
        return (question_words_weights, articles_words_weights)

    def get_weights(self, question, is_title, sum_neighbors):
        logging.info('')
        if sum_neighbors:
            logging.info('tf-idf neighbors')
        else:
            logging.info('tf-idf')

        (question_words_weights, articles_words_weight) = self.__count_tf_idf(question, is_title, sum_neighbors)
        return (question_words_weights, articles_words_weight, self._count_weights(articles_words_weight, 3))

    def upload_positions(self, question, method_name, sum_neighbors, articles_words_weight, articles_weight):
        positions = self._count_positions(question, articles_words_weight, articles_weight, True, Article.objects, Word.objects)
        if sum_neighbors:
            self._upload_positions(positions, method_name + ", type: tf_idf")
        else:
            self._upload_positions(positions, method_name + ", type: tf_idf_neighbours")
        return positions
