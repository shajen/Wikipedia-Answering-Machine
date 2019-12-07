from collections import defaultdict
from data.models import *
from functools import reduce
from termcolor import colored
import gensim.models
import logging
import numpy as np
import scipy.spatial

class Word2VecWeightCalculator():
    def __init__(self, debug_top_items, model_file):
        self.__debug_top_items = debug_top_items
        logging.info('loading word2vec model')
        self.__word2vec_model = gensim.models.KeyedVectors.load(model_file)
        logging.info('start reading stop words')
        self.__stop_words = set(Word.objects.filter(is_stop_word=True).values_list('id', flat=True))
        logging.info('stop words: %d' % len(self.__stop_words))

    def __get_similar_words_id(self, ids, topn):
        similar_words = []
        for word in Word.objects.filter(id__in=ids).values_list('value', flat=True):
            try:
                logging.debug(word)
                for (similar_word, distance) in self.__word2vec_model.most_similar(word, topn=topn):
                    logging.debug('  %s %.6f' % (similar_word, distance))
                    similar_words.append(similar_word)
            except KeyError:
                pass
        return list(Word.objects.filter(value__in=similar_words).values_list('id', flat=True)) + ids

    def __get_words_changed_forms_id(self, ids):
        return list(WordForm.objects.filter(base_word_id__in=ids).values_list('changed_word_id', flat=True)) + ids

    def __filter_stop_words_id(self, ids):
        return list(filter(lambda id: id not in self.__stop_words, ids))

    def __words_id_to_words_base_forms_id(self, ids):
        ids = list(filter(lambda id: id in self.__word_id_to_word_base_form_id, ids))
        ids = list(map(lambda id: self.__word_id_to_word_base_form_id[id], ids))
        ids = list(reduce(lambda x, y: x + y, ids, []))
        ids = self.__filter_stop_words_id(ids)
        return ids

    def __words_id_to_data(self, ids):
        ids = list(filter(lambda id: id in self.__word_id_to_vector, ids))
        data = list(map(lambda id: self.__word_id_to_vector[id], ids))
        if data:
            return np.average(data, axis=0)
        else:
            return np.zeros(100)

    def __get_question_words_id(self, question):
        return list(QuestionOccurrence.objects.filter(question_id=question.id).values_list('word_id', flat=True))

    def __get_articles_words_id(self, words_id, is_title):
        articles_words_id = defaultdict(list)
        for (article_id, word_id) in ArticleOccurrence.objects.filter(is_title=is_title, word_id__in=words_id).values_list('article_id', 'word_id'):
            articles_words_id[article_id].append(word_id)
        return articles_words_id

    def __load_words_id_vectors(self, ids):
        self.__word_id_to_vector = {}
        for (id, word) in Word.objects.filter(is_stop_word=False).filter(id__in=ids).values_list('id', 'value'):
            try:
                self.__word_id_to_vector[id] = self.__word2vec_model.get_vector(word)
            except KeyError:
                pass

    def __load_words_id_base_form_id(self, ids):
        self.__word_id_to_word_base_form_id = defaultdict(list)
        for (base_word_id, changed_word_id) in WordForm.objects.filter(changed_word_id__in=ids).values_list('base_word_id', 'changed_word_id'):
            self.__word_id_to_word_base_form_id[changed_word_id].append(base_word_id)
        for id in ids:
            self.__word_id_to_word_base_form_id[id].append(id)

    def __prepare_data(self, question, is_title, topn):
        question_words_id = self.__get_question_words_id(question)
        question_words_id = self.__filter_stop_words_id(question_words_id)
        logging.info('words count: %d' % len(set(question_words_id)))

        self.__load_words_id_base_form_id(question_words_id)
        question_words_id = self.__words_id_to_words_base_forms_id(question_words_id)
        logging.info('words base forms count: %d' % len(set(question_words_id)))

        similar_words_id = self.__get_similar_words_id(question_words_id, topn)
        logging.info('similar words count: %d' % len(set(similar_words_id)))

        similar_words_id = self.__get_words_changed_forms_id(similar_words_id)
        logging.info('changed form similar words count: %d' % len(set(similar_words_id)))

        self.__load_words_id_vectors(similar_words_id)
        logging.info('words vectors: %d' % len(set(self.__word_id_to_vector)))
        self.__load_words_id_base_form_id(similar_words_id)
        logging.info('words base forms: %d' % len(set(self.__word_id_to_word_base_form_id)))

        logging.info('reading articles')
        articles_words_id = self.__get_articles_words_id(similar_words_id, is_title)
        articles_id = []
        articles_data = []
        logging.info('preparing articles data')
        for article_id in articles_words_id:
            articles_id.append(article_id)
            articles_data.append(self.__words_id_to_data(self.__words_id_to_words_base_forms_id(articles_words_id[article_id])))
        articles_data = np.array(articles_data)
        logging.info('articles size: %s' % str(articles_data.shape))

        question_data = np.array(self.__words_id_to_data(question_words_id))
        return (question_data, articles_id, articles_data)

    def __calculate_distances(self, question_data, articles_data):
        return scipy.spatial.distance.cdist(np.array([question_data]), articles_data, 'cosine')[0]

    def __colored(self, text, colour):
        return colored(text, colour, attrs={'bold'})

    def __print(self, corrected_articles_id, position, articles_id, scores, distances):
        i = np.where(scores == position)[0][0]
        distance = distances[i]
        article = Article.objects.get(id=articles_id[i])
        colour = 'green' if articles_id[i] in corrected_articles_id else 'red'
        sign = '*' if articles_id[i] in corrected_articles_id else ' '
        logging.warning(self.__colored(' %sposition: %6d, distance: %5.4f, article: %s' % (sign, position+1, distance, article), colour))

    def __upload_positions(self, question, articles_id, distances, method):
        scores = np.argsort(np.argsort(distances))
        corrected_articles_id = list(question.answer_set.all().values_list('article_id', flat=True))
        for position in range(0, self.__debug_top_items):
            self.__print(corrected_articles_id, position, articles_id, scores, distances)

        for answer in question.answer_set.all():
            try:
                i = articles_id.index(answer.article_id)
                position = scores[i]
                if (position >= self.__debug_top_items):
                    self.__print(corrected_articles_id, position, articles_id, scores, distances)
                Solution.objects.create(position=position+1, answer=answer, method=method)
            except:
                position = 10**9
                logging.warning(self.__colored(' %sposition: %6d, distance: %5.4f, article: %s' % ('*', position, 99.99, answer.article), 'green'))
                Solution.objects.create(position=position, answer=answer, method=method)
        logging.info('')

    def has_already_solutions(self, question, method_name):
        try:
            answers = Answer.objects.filter(question=question).values_list('id', flat=True)
            method = Method.objects.get(name=method_name)
            return Solution.objects.filter(method=method, answer_id__in=answers).count() == len(answers)
        except Exception as e:
            return False

    def calculate(self, question, method_name, is_title, topn):
        logging.info('calculating')
        method, created = Method.objects.get_or_create(name=method_name)
        logging.warning(self.__colored('question: %s' % (question), 'yellow'))
        (question_data, articles_id, articles_data) = self.__prepare_data(question, is_title, topn)
        distances = self.__calculate_distances(question_data, articles_data)
        self.__upload_positions(question, articles_id, distances, method)
