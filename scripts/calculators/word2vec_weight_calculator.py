from data.models import *
import gensim.models
import logging
import numpy as np
import scipy.spatial
from tools.results_presenter import ResultsPresenter

class Word2VecModel():
    def  __init__(self, word2vec_file):
        logging.info('loading word2vec model')
        self.__word2vec_model = gensim.models.KeyedVectors.load(word2vec_file)

    def most_similar(self, word, **kwargs):
        return self.__word2vec_model.most_similar(word, **kwargs)

    def get_vector(self, word):
        return self.__word2vec_model.get_vector(word)

class Word2VecWeightCalculator():
    def __init__(self, debug_top_items, data_loader):
        self.__debug_top_items = debug_top_items
        self.__data_loader = data_loader

    def __mean(self, data):
        if np.isnan(data).all():
            return np.zeros(shape=(data.shape[1]))
        else:
            return np.nanmean(data, axis=0)

    def __prepare_question(self, question):
        return self.__mean(self.__data_loader.get_words_data(self.__data_loader.get_question_words_id(question.id)))

    def __prepare_articles(self, question, is_title, topn):
        logging.info('preparing articles data')
        question_words = set(self.__data_loader.get_question_words_id(question.id))
        if 0 in question_words:
            question_words.remove(0)
        logging.info('words set count: %d' % len(question_words))
        question_words = self.__data_loader.get_words_base_forms(question_words)
        logging.info('words set count: %d' % len(question_words))
        question_words = self.__data_loader.get_words_similar_words(question_words, topn)
        logging.info('words set count: %d' % len(question_words))
        question_words = self.__data_loader.get_words_changed_forms(question_words)
        logging.info('words set count: %d' % len(question_words))
        question_words = np.array(list(question_words))
        articles_id = []
        articles_data = []
        for article_id in self.__data_loader.get_articles_id():
            words = self.__data_loader.get_article_words_id(article_id, is_title)
            if topn < 999:
                words = np.intersect1d(words, question_words)
            if words.size:
                data = self.__data_loader.get_words_data(words)
                if not np.isnan(data).all():
                    articles_id.append(article_id)
                    articles_data.append(np.nanmean(data, axis=0))
        articles_data = np.array(articles_data)
        logging.info('articles size: %s' % str(articles_data.shape))
        return (articles_id, articles_data)

    def __calculate_distances(self, question_data, articles_data):
        if articles_data.size:
            return scipy.spatial.distance.cdist(np.array([question_data]), articles_data, 'cosine')[0]
        else:
            return np.zeros(shape=(0,1))

    def calculate(self, question, method_name, is_title, topn):
        logging.info('calculating')
        method, created = Method.objects.get_or_create(name=method_name)
        question_data = self.__prepare_question(question)
        if topn >= 999:
            try:
                articles_id = self.__articles_id
                articles_data = self.__articles_data
            except:
                (articles_id, articles_data) = self.__prepare_articles(question, is_title, topn)
                self.__articles_id = articles_id
                self.__articles_data = articles_data
        else:
            (articles_id, articles_data) = self.__prepare_articles(question, is_title, topn)
        distances = self.__calculate_distances(question_data, articles_data)
        ResultsPresenter.present(question, articles_id, distances, method, self.__debug_top_items, True)
