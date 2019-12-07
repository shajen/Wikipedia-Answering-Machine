from collections import defaultdict
from data.models import *
from functools import reduce
from termcolor import colored
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

    def __prepare_vectros(self, questions, articles, is_title):
        logging.info('preparing questions')

        questions_id = []
        questions_vectors = []
        for question in questions:
            questions_id.append(question.id)
            questions_vectors.append(self.__words_id_to_vector(self.__get_question_words(question.id)))
        questions_vectors = np.array(questions_vectors)
        logging.info('questions size: %s' % str(questions_vectors.shape))

        logging.info('preparing articles')
        articles_id = []
        articles_vectors = []
        i = 0
        for article in articles:
            i += 1
            if i % 10000 == 0:
                logging.info('iteration #%d' % i)
            articles_id.append(article.id)
            articles_vectors.append(self.__words_id_to_vector(self.__get_article_words(article.id, is_title)))
        articles_vectors = np.array(articles_vectors)
        logging.info('articles size: %s' % str(articles_vectors.shape))

        return (questions_id, questions_vectors, articles_id, articles_vectors)

    def __calculate_distances(self, question_vector, articles_vectors):
        return scipy.spatial.distance.cdist(np.array([question_vector]), articles_vectors, 'cosine')[0]

    def __colored(self, text, colour):
        return colored(text, colour, attrs={'bold'})

    def __print(self, corrected_articles_id, position, articles_id, scores, distances):
        i = np.where(scores == position)[0][0]
        distance = distances[i]
        article = Article.objects.get(id=articles_id[i])
        colour = 'green' if articles_id[i] in corrected_articles_id else 'red'
        sign = '*' if articles_id[i] in corrected_articles_id else ' '
        logging.debug(self.__colored(' %sposition: %5d, distance: %f, article: %s' % (sign, position+1, distance, article), colour))

    def __upload_positions(self, question_id, articles_id, distances, method):
        question = Question.objects.get(id=question_id)
        logging.debug(self.__colored('uploading positions, question: %s' % (question), 'yellow'))
        scores = np.argsort(np.argsort(distances))
        corrected_articles_id = list(question.answer_set.all().values_list('article_id', flat=True))
        for position in range(0, self.debug_top_items):
            self.__print(corrected_articles_id, position, articles_id, scores, distances)

        for answer in question.answer_set.all():
            i = articles_id.index(answer.article_id)
            position = scores[i]
            if (position >= self.debug_top_items):
                self.__print(corrected_articles_id, position, articles_id, scores, distances)
            Solution.objects.create(position=position+1, answer=answer, method=method)
        logging.debug('')

    def calculate(self, questions, method_name, is_title):
        articles = Article.objects.all()
        (questions_id, questions_vectors, articles_id, articles_vectors) = self.__prepare_vectros(questions, articles, is_title)
        method, created = Method.objects.get_or_create(name=('%s, type: word2vec' % method_name))
        for i in range(0, len(questions_id)):
            distances = self.__calculate_distances(questions_vectors[i], articles_vectors)
            self.__upload_positions(questions_id[i], articles_id, distances, method)
