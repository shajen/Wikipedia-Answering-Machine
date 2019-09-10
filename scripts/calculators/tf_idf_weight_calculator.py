from collections import defaultdict, deque, Counter
from data.models import *
from functools import reduce
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

    def __count_tf_idf(self, question, is_title, sum_neighbors, minimal_word_idf, power_factor):
        articles_words_weights = defaultdict(defaultdict)
        articles_weight = defaultdict()

        for item_id in self.articles_words_count:
            max_weight = 0.0
            counter = defaultdict(lambda: 0)
            words_positions = sorted(self.articles_positions[item_id], key=lambda d: d[0])
            words_positions = filter(lambda data: self.words_idf[data[1]] > minimal_word_idf, words_positions)
            current_words = deque()
            for data in words_positions:
                current_words.append(data)
                counter[data[1]] += 1
                while current_words[0][0] + sum_neighbors < current_words[-1][0]:
                    pop_data = current_words.popleft()
                    counter[pop_data[1]] -= 1
                    if counter[pop_data[1]] == 0:
                        del counter[pop_data[1]]

                weight = 0.0
                for word_id, count in counter.items():
                    weight += count / (sum_neighbors + 1) * self.words_idf[word_id]

                weight *= math.pow(len(counter), power_factor)
                if weight > max_weight:
                    max_weight = weight
                    for word_id, count in counter.items():
                        articles_words_weights[item_id][word_id = count / (sum_neighbors + 1) * self.words_idf[word_id]
                    articles_weight[item_id] = weight

        return (articles_words_weights, articles_weight)

    def get_weights(self, question, is_title, sum_neighbors):
        logging.info('')
        logging.info('tf-idf %d neighbors' % sum_neighbors)

        (articles_words_weight, articles_weight) = self.__count_tf_idf(question, is_title, sum_neighbors, 0.0, 0)
        return (self.question_words_weights, articles_words_weight, articles_weight)

    def upload_positions(self, question, method_name, sum_neighbors, articles_words_weight, articles_weight):
        positions = self._count_positions(question, articles_words_weight, articles_weight, True, Article.objects, Word.objects)
        self._upload_positions(positions, "%s, type: tf_idf_%03d_neighbors" % (method_name, sum_neighbors))
        return positions
