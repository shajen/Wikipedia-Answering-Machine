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
        logging.info('finish reading articles')

    def print_article(self, article_id, is_title, articles_words_count, articles_weight, articles_words_weights):
        try:
            weight = articles_weight[article_id]
        except:
            weight = 0.0

        if is_title:
            logging.info('%7d: %3.6f, %d, %s' % (article_id, weight, self.articles_title_count[article_id], self.articles_title[article_id]))
        else:
            logging.info('%7d: %3.6f, %d, %s' % (article_id, weight, self.articles_content_count[article_id], self.articles_title[article_id]))

        for word_id in articles_words_weights[article_id]:
            word = Word.objects.get(id=word_id)
            weight = articles_words_weights[article_id][word_id]
            logging.debug('  - %8d: %3.6f, %d, %s' % (word_id, weight, articles_words_count[article_id][word_id], word.changed_form))

    def count_tf_idf(self, question_id, is_title):
        logging.info('')
        question = Question.objects.get(id=question_id)
        logging.info('processing question:')
        logging.info('%d: %s' % (question.id, question.name))
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

        articles_weight = {}
        for article_id in articles_words_weights:
            count = len(articles_words_weights[article_id])
            articles_weight[article_id] = math.pow(count, 3) * reduce((lambda x, y: x + y), articles_words_weights[article_id].values())
        articles_ranking = list(map(lambda x: x[0], sorted(articles_weight.items(), key=operator.itemgetter(1), reverse=True)))

        if self.debug_top_articles > 0:
            logging.info('top %d articles:' % self.debug_top_articles)
            for article_id in articles_ranking[:self.debug_top_articles]:
                self.print_article(article_id, is_title, articles_words_count, articles_weight, articles_words_weights)

        logging.info('expected articles:')
        for answer in question.answer_set.all():
            self.print_article(answer.article.id, is_title, articles_words_count, articles_weight, articles_words_weights)
