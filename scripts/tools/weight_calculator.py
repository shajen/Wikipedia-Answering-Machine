from collections import defaultdict
from data.models import *
from functools import reduce
import logging
import math
import operator
import re

class WeightCalculator:
    def __init__(self, debug_top_articles):
        logging.info('start reading articles')
        self.debug_top_articles = debug_top_articles
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

    def print_article(self, article_id, articles_weight, articles_id_weights):
        try:
            weight = articles_weight[article_id]
        except:
            weight = 0.0

        logging.info('%7d: %3.6f, %d, %s' % (article_id, weight, len(articles_id_weights[article_id]), self.articles_title[article_id]))

        if logging.getLogger().level <= logging.DEBUG:
            keys = list(map(lambda x: x[0], sorted(articles_id_weights[article_id].items(), key=operator.itemgetter(1), reverse=True)))
            for id in keys:
                weight = articles_id_weights[article_id][id]
                logging.debug('  - %8d: %3.6f' % (id, weight))

    def get_article_position(self, articles_ranking, article_id):
        for i in range(len(articles_ranking)):
            if articles_ranking[i][0] == article_id:
                return i + 1
        return 10**9

    def count_weights(self, articles_id_weights, power_factor):
        articles_weight = {}
        for article_id in articles_id_weights:
            count = len(articles_id_weights[article_id])
            articles_weight[article_id] = math.pow(count, power_factor) * reduce((lambda x, y: x + y), articles_id_weights[article_id].values())
        return articles_weight

    def count_positions(self, question, articles_id_weights, power_factor):
        for article_id in articles_id_weights:
            keys = list(map(lambda x: x[0], sorted(articles_id_weights[article_id].items(), key=operator.itemgetter(1), reverse=True)))[:10]
            articles_id_weights[article_id] = {k: v for k, v in articles_id_weights[article_id].items() if k in keys}

        articles_weight = self.count_weights(articles_id_weights, power_factor)
        articles_ranking = list(map(lambda x: (x[0], x[1]), sorted(articles_weight.items(), key=operator.itemgetter(1), reverse=True)))

        if self.debug_top_articles > 0:
            logging.info('top %d articles:' % self.debug_top_articles)
            for (article_id, article_weight) in articles_ranking[:self.debug_top_articles]:
                self.print_article(article_id, articles_weight, articles_id_weights)

        logging.info('expected articles:')
        answers_positions = []
        for answer in question.answer_set.all():
            self.print_article(answer.article.id, articles_weight, articles_id_weights)
            position = self.get_article_position(articles_ranking, answer.article.id)
            logging.info('position: %d' % position)
            answers_positions.append((answer, position))
        return answers_positions

    def count_tf_idf(self, question, is_title):
        words = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', question.name)
        words = Word.objects.filter(changed_form__in=words, is_stop_word=False).values('id')
        words = list(map(lambda x: x['id'], words))

        articles_words_count = defaultdict(lambda: defaultdict(lambda: 0))
        occurrences = Occurrence.objects.filter(word_id__in=words, is_title=is_title).values('id', 'article_id', 'word_id', 'positions_count')
        for occurrence in occurrences:
            articles_words_count[occurrence['article_id']][occurrence['word_id']] += occurrence['positions_count']

        words_articles_count = defaultdict(lambda: 0)
        for article_id in articles_words_count:
            for word_id in articles_words_count[article_id]:
                words_articles_count[word_id] += 1

        words_idf = {}
        for word_id in words_articles_count:
            words_idf[word_id] = math.log(len(self.articles_title_count) / words_articles_count[word_id])

        articles_words_weights = defaultdict(defaultdict)
        for article_id in articles_words_count:
            for word_id in articles_words_count[article_id]:
                if is_title:
                    tf = articles_words_count[article_id][word_id] / self.articles_title_count[article_id]
                else:
                    tf = articles_words_count[article_id][word_id] / self.articles_content_count[article_id]
                articles_words_weights[article_id][word_id] = tf * words_idf[word_id]
        return articles_words_weights

    def count_categories_weight(self, articles_weight):
        categories_articles_weight = defaultdict(defaultdict)
        for category_id in self.categories_articles:
            for article_id in self.categories_articles[category_id]:
                try:
                    categories_articles_weight[category_id][article_id] = articles_weight[article_id]
                except Exception as e:
                    pass
        return categories_articles_weight

    def count_articles_links_weight(self, articles_weight):
        articles_links_articles_weight = defaultdict(defaultdict)
        articles_reverse_links_articles_weight = defaultdict(defaultdict)
        for article_id in self.articles_links:
            for article_link_id in self.articles_links[article_id]:
                try:
                    articles_links_articles_weight[article_id][article_link_id] = articles_weight[article_link_id]
                except Exception as e:
                    pass
                try:
                    articles_reverse_links_articles_weight[article_link_id][article_id] = articles_weight[article_id]
                except Exception as e:
                    pass
        return (articles_links_articles_weight, articles_reverse_links_articles_weight)
