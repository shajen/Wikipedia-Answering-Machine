from collections import defaultdict
from data.models import *
from functools import reduce
from collections import Counter
import logging
import math
import operator
import re
import numpy as np

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

        logging.info('start reading categories')
        self.categories_articles = defaultdict(set)
        for data in Category.objects.values('id', 'article'):
            self.categories_articles[data['id']].add(data['article'])

        logging.info('start reading links')
        self.articles_links = defaultdict(set)
        for data in Article.objects.values('id','links'):
            self.articles_links[data['id']].add(data['links'])
        logging.info('finish reading')

    def print_item(self, item_id, items_weight, items_id_weights, item_objects, id_objects):
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

    def get_article_position(self, items_ranking, item_id):
        for i in range(len(items_ranking)):
            if items_ranking[i][0] == item_id:
                return i + 1
        return 10**9

    def count_weights(self, items_id_weights, power_factor):
        items_weight = {}
        for item_id in items_id_weights:
            count = len(items_id_weights[item_id])
            items_weight[item_id] = math.pow(count, power_factor) * reduce((lambda x, y: x + y), items_id_weights[item_id].values(), 0.0)
        if items_weight:
            max_weight = max(items_weight.values())
            items_weight = {k: v / max_weight for k, v in items_weight.items()}
        return items_weight

    def count_positions(self, question, items_id_weights, power_factor, item_objects, id_objects):
        items_weight = self.count_weights(items_id_weights, power_factor)
        items_ranking = Counter(items_weight).most_common()

        if self.debug_top_items > 0:
            logging.info('top %d items:' % self.debug_top_items)
            for (item_id, weight) in items_ranking[:self.debug_top_items]:
                self.print_item(item_id, items_weight, items_id_weights, item_objects, id_objects)

        logging.info('expected items:')
        answers_positions = []
        for answer in question.answer_set.all():
            self.print_item(answer.article.id, items_weight, items_id_weights, item_objects, id_objects)
            position = self.get_article_position(items_ranking, answer.article.id)
            logging.info('position: %d' % position)
            answers_positions.append((answer, position))
        return answers_positions

    def count_tf_idf(self, question, is_title, sum_neighbors):
        words = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', question.name)
        words = Word.objects.filter(changed_form__in=words, is_stop_word=False).values('id')
        words = list(map(lambda x: x['id'], words))

        articles_words_count = defaultdict(lambda: defaultdict(lambda: 0))
        articles_words_positions = defaultdict(defaultdict)
        articles_positions = defaultdict(list)
        if sum_neighbors:
            occurrences = Occurrence.objects.filter(word_id__in=words, is_title=is_title).values('id', 'article_id', 'word_id', 'positions_count', 'positions')
        else:
            occurrences = Occurrence.objects.filter(word_id__in=words, is_title=is_title).values('id', 'article_id', 'word_id', 'positions_count')
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
        return articles_words_weights

    def get_top_n_items(self, dict, n):
        return {k: v for (k, v) in Counter(dict).most_common(n)}

    def count_articles_categories_weight(self, categories_items_weight, n, power_factor):
        for category_id in categories_items_weight:
            categories_items_weight[category_id] = self.get_top_n_items(categories_items_weight[category_id], n)
        categories_weight = self.count_weights(categories_items_weight, power_factor)

        if self.debug_top_items > 0:
            categories_ranking = Counter(categories_weight).most_common()
            logging.info('top %d categories:' % self.debug_top_items)
            for (id, weight) in categories_ranking[:self.debug_top_items]:
                self.print_item(id, categories_weight, categories_items_weight, Category.objects, Article.objects)

        articles_categories_weight = defaultdict(defaultdict)
        for category_id in self.categories_articles:
            for article_id in self.categories_articles[category_id]:
                articles_categories_weight[article_id][category_id] = categories_weight[category_id]
        for article_id in articles_categories_weight:
            articles_categories_weight[article_id] = self.get_top_n_items(articles_categories_weight[article_id], n)
        return articles_categories_weight

    def count_categories_weight(self, articles_weight, n):
        categories_items_weight = defaultdict(defaultdict)
        for category_id in self.categories_articles:
            for article_id in self.categories_articles[category_id]:
                if article_id in articles_weight:
                    categories_items_weight[category_id][article_id] = articles_weight[article_id]
            categories_items_weight[category_id] = self.get_top_n_items(categories_items_weight[category_id], n)
        return categories_items_weight

    def count_articles_links_weight(self, articles_weight, n):
        articles_links_items_weight = defaultdict(defaultdict)
        articles_reverse_links_items_weight = defaultdict(defaultdict)
        for article_id in self.articles_links:
            for article_link_id in self.articles_links[article_id]:
                if article_id and article_link_id:
                    if article_link_id in articles_weight:
                        articles_links_items_weight[article_id][article_link_id] = articles_weight[article_link_id]
                    if article_id in articles_weight:
                        articles_reverse_links_items_weight[article_link_id][article_id] = articles_weight[article_id]
        for article_id in articles_links_items_weight:
            articles_links_items_weight[article_id] = self.get_top_n_items(articles_links_items_weight[article_id], n)
        for article_id in articles_reverse_links_items_weight:
            articles_reverse_links_items_weight[article_id] = self.get_top_n_items(articles_reverse_links_items_weight[article_id], n)
        return (articles_links_items_weight, articles_reverse_links_items_weight)
