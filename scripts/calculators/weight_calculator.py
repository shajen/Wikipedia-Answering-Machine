from collections import Counter
from data.models import *
from functools import reduce
import logging
import math
import operator

class WeightCalculator:
    def __init__(self, debug_top_items):
        self.debug_top_items = debug_top_items
        logging.info('start reading articles')
        self.articles_count = Article.objects.filter(content_words_count__gte=20).count()
        logging.info('finish reading')

    def __print_item(self, item_id, items_weight, items_id_weights, item_objects, id_objects):
        try:
            weight = items_weight[item_id]
        except:
            weight = 0.0

        logging.info('%7d: %3.6f, %d, %s' % (item_id, weight, len(items_id_weights[item_id]), item_objects.get(id=item_id)))

        if logging.getLogger().level <= logging.DEBUG:
            keys = list(map(lambda x: x[0], sorted(items_id_weights[item_id].items(), key=operator.itemgetter(1), reverse=True)))
            for id in keys:
                weight = items_id_weights[item_id][id]
                id_string = ', '.join(list(map(lambda x: str(id_objects.get(id=x)), list(id))))
                logging.debug('  - %-40s - %3.6f' % (id_string, weight))

    def __get_article_position(self, items_ranking, item_id):
        for i in range(len(items_ranking)):
            if items_ranking[i][0] == item_id:
                return i + 1
        return 10**9

    def _count_positions(self, question, items_id_weights, items_weight, ascending_order, item_objects, id_objects):
        items_ranking = sorted(items_weight.items(), key=operator.itemgetter(1), reverse=ascending_order)

        if self.debug_top_items > 0:
            logging.info('top %d items:' % self.debug_top_items)
            for (item_id, weight) in items_ranking[:self.debug_top_items]:
                self.__print_item(item_id, items_weight, items_id_weights, item_objects, id_objects)

        logging.info('expected items:')
        answers_positions = []
        for answer in question.answer_set.all():
            self.__print_item(answer.article.id, items_weight, items_id_weights, item_objects, id_objects)
            position = self.__get_article_position(items_ranking, answer.article.id)
            logging.info('position: %d' % position)
            answers_positions.append((answer, position))
        return answers_positions

    def _upload_positions(self, positions, method_name):
        logging.info(positions)
        method, created = Method.objects.get_or_create(name=method_name)
        for (answer, position) in positions:
            Solution.objects.create(answer=answer, position=position, method=method)
