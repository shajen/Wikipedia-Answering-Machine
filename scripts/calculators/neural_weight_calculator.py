from collections import defaultdict
from data.models import *
from functools import reduce
import gensim.models
import logging
import numpy as np
import sys
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

    def __load_file(self, name):
        filename = '%s/%s.npy' % (self.__workdir, name)
        logging.debug("loading array from file: %s" % filename)
        data = np.load(filename)
        logging.debug("data size: %s" % str(data.shape))
        return data

    def prepare_data(self, questions, questionWords, articleTitleWords, articleContentWords, goodBadArticlesRatio):
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
        self.__prepareArticle(good_articles_id, articleContentWords, 'good_articles_content_data', False)
        self.__prepareArticle(bad_articles_id, articleTitleWords, 'bad_articles_title_data', True)
        self.__prepareArticle(bad_articles_id, articleContentWords, 'bad_articles_content_data', False)
        good_questions_data = self.__prepareQuestion(questions_id, questionWords)
        good_target = np.array([1.0] * good_questions_data.shape[0])
        bad_questions_data = np.random.permutation(np.repeat(good_questions_data, goodBadArticlesRatio, axis=0))
        bad_target = np.array([0.0] * bad_questions_data.shape[0])

        self.__save_file('good_questions_data', good_questions_data)
        self.__save_file('good_target', good_target)
        self.__save_file('bad_questions_data', bad_questions_data)
        self.__save_file('bad_target', bad_target)

    def __prepare_dataset(self, train_data_percentage):
        good_questions_data = self.__load_file('good_questions_data')
        good_articles_title_data = self.__load_file('good_articles_title_data')
        good_articles_content_data = self.__load_file('good_articles_content_data')
        good_target = self.__load_file('good_target')

        bad_questions_data = self.__load_file('bad_questions_data')
        bad_articles_title_data = self.__load_file('bad_articles_title_data')
        bad_articles_content_data = self.__load_file('bad_articles_content_data')
        bad_target = self.__load_file('bad_target')

        order = np.random.permutation(good_questions_data.shape[0] + bad_questions_data.shape[0])
        questions = np.concatenate((good_questions_data, bad_questions_data))[order]
        articles_title = np.concatenate((good_articles_title_data, bad_articles_title_data))[order]
        articles_content = np.concatenate((good_articles_content_data, bad_articles_content_data))[order]
        target = np.concatenate((good_target, bad_target))[order]

        split_index = int(questions.shape[0] * train_data_percentage)
        return ((questions[:split_index], articles_title[:split_index], articles_content[:split_index], target[:split_index]),
                (questions[split_index:], articles_title[split_index:], articles_content[split_index:], target[split_index:]))

    def __test_model(self, model, questions, articles_title, articles_content, target):
        test_scores = model.evaluate(
            { 'questions': questions, 'articles_title': articles_title, 'articles_content': articles_content },
            { 'weight': target },
            verbose = 0)
        logging.info('test dataset loss: %.4f' % test_scores[0])
        logging.info('test dataset accuracy: %.4f' % test_scores[1])

    def __create_model(self, questions_size, articles_title_size, articles_content_size):
        filters = 20
        w2c = 100
        questions_data_input = tf.keras.Input(shape=(questions_size, w2c), name='questions')
        questions_block = tf.keras.layers.Conv1D(filters, 5, activation='relu')(questions_data_input)
        questions_block = tf.keras.layers.AveragePooling1D(2)(questions_block)
        questions_block = tf.keras.layers.Conv1D(filters, 4, activation='relu')(questions_block)

        articles_title_data_input = tf.keras.Input(shape=(articles_title_size, w2c), name='articles_title')
        articles_title_block = tf.keras.layers.Conv1D(filters, 5, activation='relu')(articles_title_data_input)
        articles_title_block = tf.keras.layers.AveragePooling1D(2)(articles_title_block)
        articles_title_block = tf.keras.layers.Conv1D(filters, 4, activation='relu')(articles_title_block)

        articles_content_data_input = tf.keras.Input(shape=(articles_content_size, w2c), name='articles_content')
        articles_content_block = tf.keras.layers.Conv1D(filters, 5, activation='relu')(articles_content_data_input)
        articles_content_block = tf.keras.layers.AveragePooling1D(2)(articles_content_block)
        articles_content_block = tf.keras.layers.Conv1D(filters, 5, activation='relu')(articles_content_block)
        articles_content_block = tf.keras.layers.AveragePooling1D(2)(articles_content_block)
        articles_content_block = tf.keras.layers.Conv1D(filters, 5, activation='relu')(articles_content_block)
        articles_content_block = tf.keras.layers.AveragePooling1D(2)(articles_content_block)
        articles_content_block = tf.keras.layers.Conv1D(filters, 5, activation='relu')(articles_content_block)

        x = tf.keras.layers.add([questions_block, articles_title_block, articles_content_block])
        x = tf.keras.layers.Conv1D(filters, 2, activation='relu')(x)
        x = tf.keras.layers.Conv1D(filters, 2, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(20, activation='sigmoid')(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid', name='weight')(x)

        model = tf.keras.Model(inputs=[questions_data_input, articles_title_data_input, articles_content_data_input], outputs=x, name='wiki_qa')
        tf.keras.utils.plot_model(model, '%s/model.png' % self.__workdir, show_shapes=True)
        return model

    def train(self, use_last_trained, epoch, train_data_percentage):
        logging.info('training model')
        (train_data, test_data) = self.__prepare_dataset(train_data_percentage)
        (train_questions, train_articles_title, train_articles_content, train_target) = train_data
        (test_questions, test_articles_title, test_articles_content, test_target) = test_data
        logging.debug("train dataset:")
        logging.debug("questions: %s" % str(train_questions.shape))
        logging.debug("articles_title: %s" % str(train_articles_title.shape))
        logging.debug("articles_content: %s" % str(train_articles_content.shape))
        logging.debug("target: %s" % str(train_target.shape))

        logging.debug("train dataset:")
        logging.debug("questions: %s" % str(test_questions.shape))
        logging.debug("articles_title: %s" % str(test_articles_title.shape))
        logging.debug("articles_content: %s" % str(test_articles_content.shape))
        logging.debug("target: %s" % str(test_target.shape))

        if use_last_trained:
            logging.info('use last trained model')
            model = tf.keras.models.load_model('%s/model.h5' % self.__workdir)
            self.__test_model(model, test_questions, test_articles_title, test_articles_content, test_target)
        else:
            logging.info('create model')
            model = self.__create_model(train_questions.shape[1], train_articles_title.shape[1], train_articles_content.shape[1])
        logging.info('trainable weights: %s' % model.count_params())

        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath='%s/model.h5' % self.__workdir, save_best_only=True, verbose=0)
        model.compile(
            optimizer = tf.keras.optimizers.RMSprop(1e-3),
            loss = tf.keras.losses.binary_crossentropy,
            loss_weights = [0.2],
            metrics = ['accuracy'])
        try:
            model.fit(
                { 'questions': train_questions, 'articles_title': train_articles_title, 'articles_content': train_articles_content },
                { 'weight': train_target },
                batch_size = 64,
                epochs = epoch,
                validation_split = 0.2,
                verbose = 1,
                callbacks=[save_callback])
        except KeyboardInterrupt:
            print('\n', file=sys.stderr)
            logging.info('learing stoppped by user')

        self.__test_model(model, test_questions, test_articles_title, test_articles_content, test_target)
