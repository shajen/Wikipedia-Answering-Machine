from collections import defaultdict
from data.models import *
from functools import reduce
import calculators.weight_calculator
import gensim.models
import logging
import math
import numpy as np
import re
import scipy.spatial

class Word2VecWeightCalculator(calculators.weight_calculator.WeightCalculator):
    def __init__(self, debug_top_items, model_file):
        super().__init__(debug_top_items)

        logging.info('loading word2vec model')
        self.__word2vec_model = gensim.models.KeyedVectors.load(model_file)

        logging.info('start reading stop words')
        self.__stop_words = set(Word.objects.filter(is_stop_word=True).values_list('id', flat=True))
        logging.info('stop words: %d' % len(self.__stop_words))

        logging.info('start reading words base forms')
        self.__word_id_to_word_base_form_id = defaultdict(list)
        for wordForm in WordForm.objects.all().values('base_word_id', 'changed_word_id'):
            self.__word_id_to_word_base_form_id[wordForm['changed_word_id']].append(wordForm['base_word_id'])
        logging.info('words base forms: %d' % len(self.__word_id_to_word_base_form_id))

        logging.info('start reading words vectors')
        self.__word_id_to_vector = {}
        for word in Word.objects.filter(is_stop_word=False).values('id', 'value'):
            try:
                self.__word_id_to_vector[word['id']] = self.__word2vec_model.get_vector(word['value'])
            except KeyError:
                pass
        logging.info('words vectors: %d' % len(self.__word_id_to_vector))

        logging.info('finish')

    def __words_id_to_vector(self, ids):
        ids = list(filter(lambda id: id in self.__word_id_to_word_base_form_id, ids))
        ids = list(map(lambda id: self.__word_id_to_word_base_form_id[id], ids))
        ids = list(reduce(lambda x, y: x + y, ids, []))
        ids = list(filter(lambda id: id in self.__word_id_to_vector, ids))
        data = list(map(lambda id: self.__word_id_to_vector[id], ids))
        if data:
            return np.average(data, axis=0)
        else:
            return np.zeros(100)

    def __get_article_words(self, article_id, is_title):
        return list(ArticleOccurrence.objects.filter(article_id=article_id, is_title=is_title).values_list('word_id', flat=True))

    def __get_question_words(self, question_id):
        return list(QuestionOccurrence.objects.filter(question_id=question_id).values_list('word_id', flat=True))

    def __calculate_distances(self, questions, articles, is_title):
        logging.info('calculating distances')

        logging.info('calculating questions')
        questions_id = []
        questions_vector = []
        for question in questions:
            questions_id.append(question.id)
            questions_vector.append(self.__words_id_to_vector(self.__get_question_words(question.id)))

        logging.info('calculating articles')
        articles_id = []
        articles_vector = []
        i = 0
        for article in articles:
            i += 1
            if i % 10000 == 0:
                logging.info('iteration #%d' % i)
            articles_id.append(article.id)
            articles_vector.append(self.__words_id_to_vector(self.__get_article_words(article.id, is_title)))

        articles_vector = np.array(articles_vector)
        questions_vector = np.array(questions_vector)
        logging.info('questions size: %s' % str(questions_vector.shape))
        logging.info('articles size: %s' % str(articles_vector.shape))
        return (questions_id, articles_id, scipy.spatial.distance.cdist(questions_vector, articles_vector, 'cosine'))

    def __upload_positions(self, questions_id, articles_id, distances, method_name):
        logging.info('uploading positions')
        logging.info('distances size: %s' % str(distances.shape))
        method, created = Method.objects.get_or_create(name=('%s, type: word2vec' % method_name))
        scores = np.argsort(distances, axis=1)
        for i in range(0, len(questions_id)):
            question = Question.objects.get(id=questions_id[i])
            logging.debug(question)
            for answer in question.answer_set.all():
                try:
                    j = articles_id.index(answer.article_id)
                    position = scores[i][j] + 1
                    logging.debug('  article:  %s' % answer.article)
                    logging.debug('  position: %d' % position)
                    logging.debug('  distance: %.6f' % distances[i][j])
                    Solution.objects.create(position=position, answer=answer, method=method)
                except ValueError:
                    pass
            logging.debug('')

    def calculate(self, questions, articles, method_name, is_title):
        (questions_id, articles_id, distances) = self.__calculate_distances(questions, articles, is_title)
        self.__upload_positions(questions_id, articles_id, distances, method_name)
