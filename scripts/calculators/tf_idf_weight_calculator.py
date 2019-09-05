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
        logging.info('start parsing questions')
        self.questions_words_count = defaultdict(lambda: 0)
        for question in Question.objects.all():
            for word in self.__parse_question(question).values():
                self.questions_words_count[word] += 1
        logging.info('finish parsing questions')

    def __parse_question(self, question):
        words = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', question.name)
        words = Word.objects.filter(changed_form__in=words, is_stop_word=False).values('base_form')
        words = list(map(lambda x: x['base_form'], words))
        words = Word.objects.filter(changed_form__in=words, base_form__in=words, is_stop_word=False).values('id', 'base_form')
        base_form_to_id = {}
        for word in words:
            base_form_to_id[word['base_form']] = word['id']
        return base_form_to_id

    def prepare(self, question, is_title):
        question_words_base_form_to_id = self.__parse_question(question)
        question_words_changed_form_to_base_form = {}
        for word in Word.objects.filter(base_form__in=question_words_base_form_to_id.keys(), is_stop_word=False).values('id', 'base_form'):
            question_words_changed_form_to_base_form[word['id']] = question_words_base_form_to_id[word['base_form']]

        words_idf = {}
        for word in set(question_words_changed_form_to_base_form.values()):
            words_idf[word] = math.log(Question.objects.count() / self.questions_words_count[word])

        logging.debug('question words weights')
        self.question_words_weights = {}
        for word in set(question_words_changed_form_to_base_form.values()):
            try:
                tf = 1.0 / len(set(question_words_changed_form_to_base_form.values()))
                self.question_words_weights[word] = tf * words_idf[word]
                logging.debug(' - %-40s %.6f (%3d)' % (Word.objects.get(id=word), self.question_words_weights[word], self.questions_words_count[word]))
            except:
                pass

        self.articles_words_count = defaultdict(lambda: defaultdict(lambda: 0))
        self.articles_words_positions = defaultdict(defaultdict)
        self.articles_positions = defaultdict(list)
        occurrences = Occurrence.objects.filter(word_id__in=question_words_changed_form_to_base_form.keys(), is_title=is_title).values('article_id', 'word_id', 'positions_count', 'positions')
        for occurrence in occurrences:
            base_form_id = question_words_changed_form_to_base_form[occurrence['word_id']]
            self.articles_words_count[occurrence['article_id']][base_form_id] += occurrence['positions_count']
            #if len(positions) == occurrence['positions_count']:
            positions = [int(p) for p in occurrence['positions'].strip().split(',') if p]
            self.articles_words_positions[occurrence['article_id']][base_form_id] = positions
            self.articles_positions[occurrence['article_id']].extend([(p, base_form_id) for p in positions])

        words_articles_count = defaultdict(lambda: 0)
        for item_id in self.articles_words_count:
            for word_id in self.articles_words_count[item_id]:
                words_articles_count[word_id] += 1

        self.words_idf = {}
        for word_id in words_articles_count:
            self.words_idf[word_id] = math.log(len(self.articles_title_count) / words_articles_count[word_id])

    def __count_tf_idf(self, question, is_title, sum_neighbors):
        articles_words_weights = defaultdict(defaultdict)
        for item_id in self.articles_words_count:
            for word_id in self.articles_words_count[item_id]:
                if sum_neighbors:
                    words_positions = set()
                    for position in self.articles_words_positions[item_id][word_id]:
                        positions = [p for (p, word_id) in self.articles_positions[item_id] if abs(p - position) <= sum_neighbors]
                        words_positions.update(positions)
                    words_count = len(words_positions)
                else:
                    words_count = self.articles_words_count[item_id][word_id]

                if is_title:
                    tf = words_count / self.articles_title_count[item_id]
                else:
                    tf = words_count / self.articles_content_count[item_id]
                articles_words_weights[item_id][word_id] = tf * self.words_idf[word_id]
        return articles_words_weights

    def get_weights(self, question, is_title, sum_neighbors):
        logging.info('')
        logging.info('tf-idf %d neighbors' % sum_neighbors)

        articles_words_weight = self.__count_tf_idf(question, is_title, sum_neighbors)
        return (self.question_words_weights, articles_words_weight, self._count_weights(articles_words_weight, 3))

    def upload_positions(self, question, method_name, sum_neighbors, articles_words_weight, articles_weight):
        positions = self._count_positions(question, articles_words_weight, articles_weight, True, Article.objects, Word.objects)
        self._upload_positions(positions, "%s, type: tf_idf_%02d_neighbours" % (method_name, sum_neighbors))
        return positions
