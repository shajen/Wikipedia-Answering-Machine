from collections import Counter
from data.models import *
from functools import reduce
import logging
import math
import operator

class WeightCalculator:
    def __init__(self, debug_top_items):
        logging.info('start reading articles')
        self.debug_top_items = debug_top_items
        articles = Article.objects.values('id', 'title_words_count', 'content_words_count', 'title')
        self.articles_title_count = {}
        self.articles_content_count = {}
        self.articles_title = {}
        for article in articles:
            self.articles_title_count[article['id']] = article['title_words_count']
            self.articles_content_count[article['id']] = article['content_words_count']
            self.articles_title[article['id']] = article['title']
        logging.info('finish reading')

    def _print_item(self, item_id, items_weight, items_id_weights, item_objects, id_objects):
        try:
            weight = items_weight[item_id]
        except:
            weight = 0.0

        logging.info('%7d: %3.6f, %d, %s' % (item_id, weight, len(items_id_weights[item_id]), item_objects.get(id=item_id)))

        if logging.getLogger().level <= logging.DEBUG:
            keys = list(map(lambda x: x[0], sorted(items_id_weights[item_id].items(), key=operator.itemgetter(1), reverse=True)))
            for id in keys:
                weight = items_id_weights[item_id][id]
                logging.debug('  - %8d: %3.6f - %s' % (id, weight, id_objects.get(id=id)))

    def __get_article_position(self, items_ranking, item_id):
        for i in range(len(items_ranking)):
            if items_ranking[i][0] == item_id:
                return i + 1
        return 10**9

    def _count_weights(self, items_id_weights, power_factor):
        items_weight = {}
        for item_id in items_id_weights:
            count = len(items_id_weights[item_id])
            items_weight[item_id] = math.pow(count, power_factor) * reduce((lambda x, y: x + y), items_id_weights[item_id].values(), 0.0)
        if items_weight:
            max_weight = max(items_weight.values())
            items_weight = {k: v / max_weight for k, v in items_weight.items()}
        return items_weight

    def _count_positions(self, question, items_id_weights, items_weight, ascending_order, item_objects, id_objects):
        items_ranking = sorted(items_weight.items(), key=operator.itemgetter(1), reverse=ascending_order)

        if self.debug_top_items > 0:
            logging.info('top %d items:' % self.debug_top_items)
            for (item_id, weight) in items_ranking[:self.debug_top_items]:
                self._print_item(item_id, items_weight, items_id_weights, item_objects, id_objects)

        logging.info('expected items:')
        answers_positions = []
        for answer in question.answer_set.all():
            self._print_item(answer.article.id, items_weight, items_id_weights, item_objects, id_objects)
            position = self.__get_article_position(items_ranking, answer.article.id)
            logging.info('position: %d' % position)
            answers_positions.append((answer, position))
        return answers_positions

    def _get_top_n_items(self, dict, n):
        return {k: v for (k, v) in Counter(dict).most_common(n)}

    def _upload_positions(self, positions, method_name):
        logging.info(positions)
        method, created = Method.objects.get_or_create(name=method_name)
        for (answer, position) in positions:
            Solution.objects.create(answer=answer, position=position, method=method)
