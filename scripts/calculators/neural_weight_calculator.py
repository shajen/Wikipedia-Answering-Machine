from collections import defaultdict
from data.models import *
from functools import reduce
import gensim.models
import logging
import numpy as np
import tensorflow as tf

class NeuralWeightCalculator():
    def __init__(self, debug_top_items, model_file):
        self.__debug_top_items = debug_top_items

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

    def __prepareArticle(self, articles_id, words, is_title):
        logging.debug('preparing %d articles, top words: %d, title: %d' %(len(articles_id), words, is_title))
        articles_words = defaultdict(lambda : [None] * words)
        for (article_id, word_id, positions) in ArticleOccurrence.objects.filter(article_id__in=articles_id, is_title=is_title).values_list('article_id', 'word_id', 'positions'):
            for position in positions.split(','):
                position = int(position)
                if (position <= words):
                    articles_words[article_id][position - 1] = word_id
        data = []
        logging.debug("words2vec")
        for article_id in articles_id:
            data.append(self.__words2vec(articles_words[article_id]))
        data = np.array(data)
        logging.debug("%s" % str(data.shape))
        return data

    def __prepareQuestion(self, questions_id, words):
        logging.debug('preparing %d questions, top words: %d' %(len(questions_id), words))
        questions_words = defaultdict(lambda : [None] * words)
        for (question_id, word_id, positions) in QuestionOccurrence.objects.filter(question_id__in=questions_id).values_list('question_id', 'word_id', 'positions'):
            for position in positions.split(','):
                position = int(position)
                if (position <= words):
                    questions_words[question_id][position - 1] = word_id
        data = []
        logging.debug("words2vec")
        for question_id in questions_id:
            data.append(self.__words2vec(questions_words[question_id]))
        data = np.array(data)
        logging.debug("%s" % str(data.shape))
        return data

    def prepareData(self, questions, questionWords, articleTitleWords, articleWords, goodBadArticlesRatio):
        logging.info('start preparing data')
        questionsData = []
        articlesTitleData = []
        articlesData = []
        articlesTarget = []

        questions_id = list(map(lambda q: [q.id] * q.answer_set.count(), questions))
        questions_id = list(reduce(lambda x, y: x + y, questions_id, []))
        logging.debug('questions: %d' % len(questions_id))

        good_articles_id = list(map(lambda q: list(map(lambda a: a.article_id, q.answer_set.all())), questions))
        good_articles_id = list(reduce(lambda x, y: x + y, good_articles_id, []))
        logging.debug('good articles: %d' % len(good_articles_id))

        bad_articles_id = list(Article.objects.exclude(id__in=good_articles_id).order_by('?').values_list('id', flat=True)[:goodBadArticlesRatio * len(good_articles_id)])
        logging.debug('bad articles: %d' % len(bad_articles_id))

        questions_data = self.__prepareQuestion(questions_id, questionWords)
        good_articles_title_data = self.__prepareArticle(good_articles_id, articleTitleWords, True)
        good_articles_data = self.__prepareArticle(good_articles_id, articleWords, False)
        bad_articles_title_data = self.__prepareArticle(bad_articles_id, articleTitleWords, True)
        bad_articles_data = self.__prepareArticle(bad_articles_id, articleWords, False)
