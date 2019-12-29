from collections import defaultdict
from data.models import *
from functools import reduce
import gensim.models
import logging
import numpy as np
import scipy.spatial
from tools.results_presenter import ResultsPresenter
from more_itertools import unique_everseen

class ArticlesData():
    def __init__(self, topn):
        if topn >= 999:
            top_words = 100
        else:
            top_words = 100000
        logging.info('loading articles words')
        stop_words = set(Word.objects.filter(is_stop_word=True).values_list('id', flat=True))
        self.__content_articles_words = {}
        self.__title_articles_words = {}
        for article in Article.objects.all():
            logging.debug('article: %d' % article.id)
            self.__content_articles_words[article.id] = article.get_words_unique(False, stop_words, top_words)
            self.__title_articles_words[article.id] = article.get_words_unique(True, stop_words, top_words)

    def get_article_words_id(self, article_id, words_id, is_title):
        if is_title:
            words = self.__title_articles_words[article_id]
            if words_id:
                return words_id.intersection(words)
            else:
                return words
        else:
            words = self.__content_articles_words[article_id]
            if words_id:
                return words_id.intersection(words)
            else:
                return words

    def get_articles_id(self):
        return self.__content_articles_words.keys()

class Word2VecModel():
    def  __init__(self, word2vec_file):
        logging.info('loading word2vec model')
        self.__word2vec_model = gensim.models.KeyedVectors.load(word2vec_file)

    def most_similar(self, word, **kwargs):
        return self.__word2vec_model.most_similar(word, **kwargs)

    def get_vector(self, word):
        return self.__word2vec_model.get_vector(word)

class Word2VecWeightCalculator():
    def __init__(self, debug_top_items, word2vec_model, articles_data):
        self.__debug_top_items = debug_top_items
        logging.info('start reading stop words')
        self.__stop_words = set(Word.objects.filter(is_stop_word=True).values_list('id', flat=True))
        logging.info('stop words: %d' % len(self.__stop_words))
        self.__word2vec_model = word2vec_model
        self.__articles_data = articles_data

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
        words = question.words.split(',')
        words = list(filter(lambda w: w != '', words))
        words = list(map(lambda w: int(w), words))
        return words

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

    def __prepare_question(self, question):
        question_words_id = self.__get_question_words_id(question)
        question_words_id = self.__filter_stop_words_id(question_words_id)
        self.__load_words_id_base_form_id(question_words_id)
        question_words_id = self.__words_id_to_words_base_forms_id(question_words_id)
        self.__load_words_id_vectors(question_words_id)
        return np.array(self.__words_id_to_data(question_words_id))

    def __prepare_articles(self, is_title, topn):
        similar_words_id = list(Word.objects.filter(is_stop_word=False).order_by('id').values_list('id', flat=True))
        logging.info('similar words count: %d' % len(similar_words_id))

        self.__load_words_id_vectors(similar_words_id)
        logging.info('words vectors: %d' % len(set(self.__word_id_to_vector)))
        self.__load_words_id_base_form_id(similar_words_id)
        logging.info('words base forms: %d' % len(set(self.__word_id_to_word_base_form_id)))

        articles_id = []
        articles_data = []
        logging.info('preparing articles data')
        for article_id in self.__articles_data.get_articles_id():
            articles_id.append(article_id)
            articles_data.append(self.__words_id_to_data(self.__words_id_to_words_base_forms_id(self.__articles_data.get_article_words_id(article_id, set(), is_title))))
        articles_data = np.array(articles_data)
        logging.info('articles size: %s' % str(articles_data.shape))
        return (articles_id, articles_data)

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

        articles_id = []
        articles_data = []
        similar_words_id = set(similar_words_id)
        logging.info('preparing articles data')
        for article_id in self.__articles_data.get_articles_id():
            articles_id.append(article_id)
            articles_data.append(self.__words_id_to_data(self.__words_id_to_words_base_forms_id(self.__articles_data.get_article_words_id(article_id, similar_words_id, is_title))))
        articles_data = np.array(articles_data)
        logging.info('articles size: %s' % str(articles_data.shape))

        question_data = np.array(self.__words_id_to_data(question_words_id))
        return (question_data, articles_id, articles_data)

    def __calculate_distances(self, question_data, articles_data):
        return scipy.spatial.distance.cdist(np.array([question_data]), articles_data, 'cosine')[0]

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
        if topn >= 999:
            try:
                articles_id = self.__articles_id
                articles_data = self.__articles_data
            except:
                (articles_id, articles_data) = self.__prepare_articles(is_title, topn)
                self.__articles_id = articles_id
                self.__articles_data = articles_data
            question_data = self.__prepare_question(question)
        else:
            (question_data, articles_id, articles_data) = self.__prepare_data(question, is_title, topn)
        if articles_data.size:
            distances = self.__calculate_distances(question_data, articles_data)
            ResultsPresenter.present(question, articles_id, distances, method, self.__debug_top_items, True)
