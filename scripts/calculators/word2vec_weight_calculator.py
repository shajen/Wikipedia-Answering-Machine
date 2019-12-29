from collections import defaultdict
from data.models import *
from functools import reduce
from termcolor import colored
import gensim.models
import logging
import numpy as np
import scipy.spatial

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
        for (article_id, content_words, title_words) in Article.objects.values_list('id', 'content_words', 'title_words'):
            logging.debug('article: %d' % article_id)
            content_words = content_words.split(',')
            content_words = content_words[:top_words]
            content_words = list(filter(lambda w: w != '', content_words))
            content_words = list(map(lambda w: int(w), content_words))
            content_words = list(filter(lambda w: w not in stop_words, content_words))
            self.__content_articles_words[article_id] = set(content_words)

            title_words = title_words.split(',')
            title_words = title_words[:top_words]
            title_words = list(filter(lambda w: w != '', title_words))
            title_words = list(map(lambda w: int(w), title_words))
            title_words = list(filter(lambda w: w not in stop_words, title_words))
            self.__title_articles_words[article_id] = set(title_words)

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

class ResultsPresenter():
    def __colored(text, colour):
        return colored(text, colour, attrs={'bold'})

    def __get_weight(weight, is_smaller_first):
        if np.isnan(weight):
            if is_smaller_first:
                return float(10**5)
            else:
                return 0.0
        else:
            return weight

    def __print(corrected_articles_id, position, articles_id, scores, distances, question, method, is_smaller_first):
        i = np.where(scores == position)[0][0]
        distance = distances[i]
        article = Article.objects.get(id=articles_id[i])
        colour = 'green' if articles_id[i] in corrected_articles_id else 'red'
        sign = '*' if articles_id[i] in corrected_articles_id else ' '
        logging.warning(ResultsPresenter.__colored(' %sposition: %6d, distance: %5.4f, article (%7d): %s' % (sign, position+1, distance, article.id, article), colour))
        return Rate(weight=ResultsPresenter.__get_weight(distance, is_smaller_first), article=article, question=question, method=method)

    def upload_positions(question, articles_id, distances, method, debug_top_items, is_smaller_first):
        rates = []
        if is_smaller_first:
            scores = np.argsort(np.argsort(distances))
        else:
            scores = np.argsort(np.argsort(distances)[::-1])
        corrected_articles_id = list(question.answer_set.all().values_list('article_id', flat=True))
        logging.warning(ResultsPresenter.__colored('question (%5d): %s' % (question.id, question), 'yellow'))
        for position in range(0, min(len(distances), debug_top_items)):
            rates.append(ResultsPresenter.__print(corrected_articles_id, position, articles_id, scores, distances, question, method, is_smaller_first))

        for answer in question.answer_set.all():
            try:
                i = articles_id.index(answer.article_id)
                position = scores[i]
                if (position >= debug_top_items):
                    rates.append(ResultsPresenter.__print(corrected_articles_id, position, articles_id, scores, distances, question, method, is_smaller_first))
                Solution.objects.create(position=position+1, answer=answer, method=method)
            except:
                position = 10**5
                distance = np.nan
                logging.warning(ResultsPresenter.__colored(' %sposition: %6d, distance: %5.4f, article (%7d): %s' % ('*', position, distance, answer.article.id, answer.article), 'green'))
                Solution.objects.create(position=position, answer=answer, method=method)
                rates.append(Rate(weight=ResultsPresenter.__get_weight(distance, is_smaller_first), question=question, article=answer.article, method=method))
        Rate.objects.bulk_create(rates, ignore_conflicts=True)
        logging.info('')

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

    def __prepare_data(self, question, is_title, topn):
        question_words_id = self.__get_question_words_id(question)
        question_words_id = self.__filter_stop_words_id(question_words_id)
        logging.info('words count: %d' % len(set(question_words_id)))

        self.__load_words_id_base_form_id(question_words_id)
        question_words_id = self.__words_id_to_words_base_forms_id(question_words_id)
        logging.info('words base forms count: %d' % len(set(question_words_id)))

        if topn >= 999:
            try:
                self.__word_id_to_vector
            except:
                similar_words_id = list(Word.objects.filter(is_stop_word=False).order_by('id').values_list('id', flat=True))
                logging.info('similar words count: %d' % len(similar_words_id))

                self.__load_words_id_vectors(similar_words_id)
                logging.info('words vectors: %d' % len(set(self.__word_id_to_vector)))
                self.__load_words_id_base_form_id(similar_words_id)
                logging.info('words base forms: %d' % len(set(self.__word_id_to_word_base_form_id)))

            similar_words_id = []
        else:
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
        (question_data, articles_id, articles_data) = self.__prepare_data(question, is_title, topn)
        if articles_data.size:
            distances = self.__calculate_distances(question_data, articles_data)
            ResultsPresenter.upload_positions(question, articles_id, distances, method, self.__debug_top_items, True)
