from collections import defaultdict
from data.models import *
import gensim.models
import logging
import numpy as np
import SharedArray

class DataLoader():
    def __init__(self, learning_model_count, classic_model_count, word2vec_file, word2vec_size):
        logging.info('data loader initializing')
        self.__word2vec_file = word2vec_file
        self.__word2vec_size = word2vec_size
        (self.__learning_model_questions_words_count, self.__learning_model_articles_title_words_count, self.__learning_model_articles_content_words_count) = learning_model_count
        (self.__classic_model_questions_words_count, self.__classic_model_articles_title_words_count, self.__classic_model_articles_content_words_count) = classic_model_count

        stop_words = set(Word.objects.filter(is_stop_word=True).values_list('id', flat=True))
        self.__load_questions(stop_words, self.__classic_model_questions_words_count)
        self.__load_articles(stop_words, self.__classic_model_articles_title_words_count, self.__classic_model_articles_content_words_count)
        self.__load_words(word2vec_size, 10)

    def word2vec_size(self):
        return self.__word2vec_size

    def questions_words_count(self):
        return self.__learning_model_questions_words_count

    def articles_title_words_count(self):
        return self.__learning_model_articles_title_words_count

    def articles_content_words_count(self):
        return self.__learning_model_articles_content_words_count

    def __load_word2vec_model(self):
        try:
            self.__word2vec_model
        except:
            logging.info('loading word2vec model')
            self.__word2vec_model = gensim.models.KeyedVectors.load(self.__word2vec_file, mmap='r')

    def __create_or_link(self, name, shape, type):
        try:
            data = SharedArray.attach("shm://%s" % name)
            return (data, True)
        except:
            data = SharedArray.create("shm://%s" % name, shape=shape, dtype=type)
            return (data, False)

    def __load_questions(self, stop_words, question_words):
        questions_count = Question.objects.count()
        max_questions_id = Question.objects.order_by('-id')[0].id + 1
        logging.info('questions count: %d' % questions_count)
        logging.info('max questions id: %d' % max_questions_id)

        (self.__questions_id, questions_id_ok) = self.__create_or_link('questions_id', (questions_count), np.uint32)
        (self.__questions_words, questions_words_ok) = self.__create_or_link('questions_words', (max_questions_id, question_words), np.uint32)

        if questions_words_ok:
            logging.info("data already exists")
            return

        self.__questions_id.fill(0)
        self.__questions_words.fill(0)

        i = 0
        logging.info("loading questions")
        for question in Question.objects.all():
            logging.debug('question: %d' % question.id)
            self.__questions_id[i] = question.id
            self.__questions_words[question.id] = question.get_words(stop_words, question_words)
            i += 1

    def __load_articles(self, stop_words, article_title_words, article_content_words):
        articles_count = Article.objects.filter(content_words_count__gte=10).count()
        max_articles_id = Article.objects.order_by('-id')[0].id + 1
        logging.info('articles count: %d' % articles_count)
        logging.info('max articles id: %d' % max_articles_id)

        (self.__articles_id, articles_id_ok) = self.__create_or_link('articles_id', (articles_count), np.uint32)
        (self.__articles_title_words, articles_title_words_ok) = self.__create_or_link('articles_title_words', (max_articles_id, article_title_words), np.uint32)
        (self.__articles_content_words, articles_content_words_ok) = self.__create_or_link('articles_content_words', (max_articles_id, article_content_words), np.uint32)

        if all([articles_id_ok, articles_title_words_ok, articles_content_words_ok]):
            logging.info("data already exists")
            return

        self.__articles_id.fill(0)
        self.__articles_title_words.fill(0)
        self.__articles_content_words.fill(0)

        i = 0
        logging.info("loading articles")
        for article in Article.objects.filter(content_words_count__gte=10):
            logging.debug('article: %d' % article.id)
            self.__articles_id[i] = article.id
            self.__articles_title_words[article.id] = article.get_words(True, stop_words, article_title_words)
            self.__articles_content_words[article.id] = article.get_words(False, stop_words, article_content_words)
            i += 1

    def __load_words(self, word2vec_size, similar_words_top_n):
        words_count = Word.objects.count()
        max_words_id = Word.objects.order_by('-id')[0].id + 1
        logging.info('words count: %d' % words_count)
        logging.info('max words id: %d' % max_words_id)

        (self.__words_to_vec, words_ok) = self.__create_or_link('words_to_vec', (max_words_id, word2vec_size), np.float32)
        if all([words_ok]):
            logging.info("data already exists")
            return

        self.__words_to_vec[:] = np.nan

        self.__load_word2vec_model()
        logging.info('loading words vectors')
        for (word_id, value) in Word.objects.filter(is_stop_word=False).values_list('id', 'value'):
            logging.debug(word_id)
            try:
                self.__words_to_vec[word_id] = self.__word2vec_model.get_vector(value)
                logging.debug("vector loaded")
            except:
                logging.debug("vector not found")

        logging.info('loading words base forms')
        words_base_forms = defaultdict(set)
        for (base_word_id, changed_word_id) in WordForm.objects.values_list('base_word_id', 'changed_word_id'):
            words_base_forms[changed_word_id].add(base_word_id)

        logging.info('calculating vectors')
        for word_id in words_base_forms:
            if np.isnan(self.__words_to_vec[word_id]).all():
                data = self.__words_to_vec[list(words_base_forms[word_id])]
                if not np.isnan(data).all():
                    logging.debug("calculating vector")
                    self.__words_to_vec[word_id] = np.nanmean(data, axis=0)
                else:
                    logging.debug("no data")
            else:
                logging.debug("vector already exists")

    def get_words_similar_words(self, words, topn):
        self.__load_word2vec_model()
        similar_words = []
        for value in Word.objects.filter(id__in=words).values_list("value", flat=True):
            if value in self.__word2vec_model:
                for (similar_word, distance) in self.__word2vec_model.most_similar(value, topn=topn):
                    similar_words.append(similar_word)
        return set(Word.objects.filter(value__in=similar_words).values_list('id', flat=True)) | words

    def get_words_base_forms(self, words):
        return set(WordForm.objects.filter(changed_word_id__in=words).values_list('base_word_id', flat=True)) | words

    def get_words_changed_forms(self, words):
        return set(WordForm.objects.filter(base_word_id__in=words).values_list('changed_word_id', flat=True)) | words

    def get_words_data(self, words):
        return self.__words_to_vec[words]

    def get_questions_id(self):
        return self.__questions_id

    def get_question_words_id(self, question_id):
        return self.__questions_words[question_id][:self.__learning_model_questions_words_count]

    def get_articles_id(self):
        return self.__articles_id

    def get_article_words_id(self, article_id, is_title):
        if is_title:
            return self.__articles_title_words[article_id][:self.__learning_model_articles_title_words_count]
        else:
            return self.__articles_content_words[article_id][:self.__learning_model_articles_content_words_count]
