from collections import defaultdict
from data.models import *
from functools import reduce
import gensim.models
import logging
import numpy as np
import tensorflow as tf

class NeuralWeightCalculator():
    def __init__(self, debug_top_items, model_file, workdir, skip_prepare_data):
        self.__debug_top_items = debug_top_items
        self.__workdir = workdir

        if skip_prepare_data:
            logging.info('skipping reading data')
            return

        logging.info('reading base forms')
        stop_words_id = list(Word.objects.filter(is_stop_word=True).values_list('id', flat=True))
        self.__changed_word_id_to_base_form_id = {}
        for (changed_word_id, base_word_id) in WordForm.objects.exclude(changed_word_id__in=stop_words_id).values_list('changed_word_id', 'base_word_id'):
            self.__changed_word_id_to_base_form_id[changed_word_id] = base_word_id

        logging.info('reading word2vec model')
        self.__word2vec_model = gensim.models.KeyedVectors.load(model_file)

        logging.info('reading words vec value')
        self.__word_id_to_vector = {}
        for (word_id, value) in Word.objects.values_list('id', 'value'):
            try:
                self.__word_id_to_vector[word_id] = self.__word2vec_model.get_vector(value)
            except KeyError:
                pass
        logging.info('words vec size: %d' % (len(self.__word_id_to_vector)))

    def __word2vec(self, word):
        if word in self.__changed_word_id_to_base_form_id:
            word = self.__changed_word_id_to_base_form_id[word]
        if word in self.__word_id_to_vector:
            return self.__word_id_to_vector[word]
        else:
            return np.array([0.0] * 100)

    def __words2vec(self, words):
        return list(map(lambda word: self.__word2vec(word), words))

    def __prepareArticle(self, articles_id, words, name, is_title):
        logging.debug('preparing %d articles, top words: %d, title: %d' %(len(articles_id), words, is_title))
        data = []
        total = len(articles_id)
        current = 0
        step = round(total / (10 if is_title else 100))
        for article_id in articles_id:
            current += 1
            if current % step == 0:
                logging.debug("progress: %d/%d (%.2f %%)" % (current, total, current / total * 100))
            article_words = [None] * words
            for (word_id, positions) in ArticleOccurrence.objects.filter(article_id=article_id, is_title=is_title).values_list('word_id', 'positions').iterator():
                for position in positions.split(','):
                    position = int(position)
                    if (position <= words):
                        article_words[position - 1] = word_id
            data.append(self.__words2vec(article_words))
        self.__save_file(name, np.array(data))

    def __prepareQuestion(self, questions_id, words):
        logging.debug('preparing %d questions, top words: %d' %(len(questions_id), words))
        data = []
        total = len(questions_id)
        current = 0
        step = round(total / 10)
        for question_id in questions_id:
            current += 1
            if (current % step == 0):
                logging.debug("progress: %d/%d (%.2f %%)" % (current, total, current / total * 100))
            question_words = [None] * words
            for (word_id, positions) in QuestionOccurrence.objects.filter(question_id=question_id).values_list('word_id', 'positions').iterator():
                for position in positions.split(','):
                    position = int(position)
                    if (position <= words):
                        question_words[position - 1] = word_id
            data.append(self.__words2vec(question_words))

        data = np.array(data)
        logging.debug("data size: %s" % str(data.shape))
        return data

    def __save_file(self, name, data):
        filename = '%s/%s.npy' % (self.__workdir, name)
        logging.debug("saving array %s to file: %s" % (str(data.shape), filename))
        np.save(filename, data)

    def prepareData(self, questions, questionWords, articleTitleWords, articleWords, goodBadArticlesRatio):
        logging.info('start preparing data')

        questions_id = list(map(lambda q: [q.id] * q.answer_set.count(), questions))
        questions_id = list(reduce(lambda x, y: x + y, questions_id, []))
        logging.debug('questions: %d' % len(questions_id))

        good_articles_id = list(map(lambda q: list(map(lambda a: a.article_id, q.answer_set.all())), questions))
        good_articles_id = list(reduce(lambda x, y: x + y, good_articles_id, []))
        logging.debug('good articles: %d' % len(good_articles_id))

        bad_articles_id = list(Article.objects.exclude(id__in=good_articles_id).order_by('?').values_list('id', flat=True)[:goodBadArticlesRatio * len(good_articles_id)])
        logging.debug('bad articles: %d' % len(bad_articles_id))

        self.__prepareArticle(good_articles_id, articleTitleWords, 'good_articles_title_data', True)
        self.__prepareArticle(good_articles_id, articleWords, 'good_articles_data', False)
        self.__prepareArticle(bad_articles_id, articleTitleWords, 'bad_articles_title_data', True)
        self.__prepareArticle(bad_articles_id, articleWords, 'bad_articles_data', False)
        good_questions_data = self.__prepareQuestion(questions_id, questionWords)
        good_target = np.array([1.0] * good_questions_data.shape[0])
        bad_questions_data = np.random.permutation(np.repeat(good_questions_data, goodBadArticlesRatio, axis=0))
        bad_target = np.array([0.0] * bad_questions_data.shape[0])

        self.__save_file('good_questions_data', good_questions_data)
        self.__save_file('good_target', good_target)
        self.__save_file('bad_questions_data', bad_questions_data)
        self.__save_file('bad_target', bad_target)
