from collections import defaultdict
from data.models import *
from functools import reduce
from termcolor import colored
from tools.results_presenter import ResultsPresenter
import gensim.models
import logging
import numpy as np
import os
import sys
import tensorflow as tf
import re

class NeuralWeightCalculator():
    _W2V_SIZE = 100
    __ARTICLES_CHUNKS = 100
    __FILTERS = 64

    def __init__(self, debug_top_items, model_file, workdir, questions_words, articles_title_words, articles_content_words, good_bad_ratio):
        self.__debug_top_items = debug_top_items
        self.__model_file = model_file
        self.__workdir = workdir
        self.__questions_words = questions_words
        self.__articles_title_words = articles_title_words
        self.__articles_content_words = articles_content_words
        self.__good_bad_ratio = good_bad_ratio
        self.__data_loaded = False
        self.__dataset_loaded = False

    def __load_data(self):
        if self.__data_loaded:
            return

        logging.info('loading data')
        logging.info('reading base forms')
        stop_words_id = list(Word.objects.filter(is_stop_word=True).values_list('id', flat=True))
        self.__changed_word_id_to_base_form_id = {}
        for (changed_word_id, base_word_id) in WordForm.objects.exclude(changed_word_id__in=stop_words_id).values_list('changed_word_id', 'base_word_id'):
            self.__changed_word_id_to_base_form_id[changed_word_id] = base_word_id

        logging.info('reading word2vec model')
        self.__word2vec_model = gensim.models.KeyedVectors.load(self.__model_file)

        logging.info('reading words vec value')
        self.__word_id_to_vector = {}
        for (word_id, value) in Word.objects.values_list('id', 'value'):
            try:
                self.__word_id_to_vector[word_id] = self.__word2vec_model.get_vector(value)
            except KeyError:
                pass
        logging.info('words vec size: %d' % (len(self.__word_id_to_vector)))
        self.__data_loaded = True

    def __colored(self, text, colour):
        return colored(text, colour, attrs={'bold'})

    def __word2vec(self, word):
        if word in self.__changed_word_id_to_base_form_id:
            word = self.__changed_word_id_to_base_form_id[word]
        if word in self.__word_id_to_vector:
            return self.__word_id_to_vector[word]
        else:
            return np.array([0.0] * NeuralWeightCalculator._W2V_SIZE)

    def __words2vec(self, words):
        return list(map(lambda word: self.__word2vec(word), words))

    def __prepare_articles(self, articles_id, top_words, is_title, show_progress):
        logging.debug('preparing %d articles, top words: %d, title: %d' %(len(articles_id), top_words, is_title))
        data = []
        total = articles_id.shape[0]
        current = 1
        step = round(total / (10 if is_title else 100))
        stop_words = set(Word.objects.filter(is_stop_word=True).values_list('id', flat=True))
        for article in Article.objects.filter(id__in=articles_id):
            words = article.get_words_keep_positions(is_title, stop_words, top_words)
            words = words +  [None] * (top_words - len(words))
            data.append(np.array(self.__words2vec(words)))
            current += 1
            if current % step == 0 and show_progress:
                logging.debug("progress: %d/%d (%.2f %%)" % (current, total, current / total * 100))
        data = np.array(data)
        logging.debug("data size: %s" % str(data.shape))
        return data

    def __prepare_questions(self, questions_id, top_words):
        logging.debug('preparing %d questions, top words: %d' %(len(questions_id), top_words))
        data = []
        total = questions_id.shape[0]
        current = 1
        step = round(total / 10)
        stop_words = set(Word.objects.filter(is_stop_word=True).values_list('id', flat=True))
        for question in Question.objects.filter(id__in=questions_id):
            words = question.get_words_keep_positions(stop_words, top_words)
            words = words +  [None] * (top_words - len(words))
            data.append(np.array(self.__words2vec(words)))
            current += 1
            if (current % step == 0):
                logging.debug("progress: %d/%d (%.2f %%)" % (current, total, current / total * 100))

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

    def __generate_dataset_with_questions(self, questions, dataset_name):
        logging.info('generating dataset: %s' % dataset_name)

        logging.debug('questions: %d' % len(questions))
        questions_id = list(map(lambda q: [q.id] * q.answer_set.count(), questions))
        questions_id = list(reduce(lambda x, y: x + y, questions_id, []))
        logging.debug('answers: %d' % len(questions_id))

        good_articles_id = list(map(lambda q: list(map(lambda a: a.article_id, q.answer_set.all())), questions))
        good_articles_id = list(reduce(lambda x, y: x + y, good_articles_id, []))
        logging.debug('good articles: %d' % len(good_articles_id))

        bad_articles_id = list(Article.objects.exclude(id__in=good_articles_id).order_by('?').values_list('id', flat=True)[:self.__good_bad_ratio * len(good_articles_id)])
        logging.debug('bad articles: %d' % len(bad_articles_id))

        articles_id = np.concatenate((np.array(good_articles_id), np.array(bad_articles_id)))
        logging.debug('total articles: %d' % len(bad_articles_id))

        questions_id = np.repeat(np.array([questions_id]), self.__good_bad_ratio + 1, axis=0).reshape(-1)
        logging.debug('total questions: %d' % len(questions_id))

        articles_title = self.__prepare_articles(articles_id, self.__articles_title_words, True, True)
        articles_content = self.__prepare_articles(articles_id, self.__articles_content_words, False, True)
        questions = self.__prepare_questions(questions_id, self.__questions_words)
        targets = np.array([1.0] * len(good_articles_id) + [0.0] * len(bad_articles_id))

        order = np.random.permutation(articles_id.shape[0])
        questions_id = questions_id[order]
        questions = questions[order]
        articles_id = articles_id[order]
        articles_title = articles_title[order]
        articles_content = articles_content[order]
        targets = targets[order]

        self.__save_file('%s_questions_id' % dataset_name, questions_id)
        self.__save_file('%s_questions' % dataset_name, questions)
        self.__save_file('%s_articles_id' % dataset_name, articles_id)
        self.__save_file('%s_articles_title' % dataset_name, articles_title)
        self.__save_file('%s_articles_content' % dataset_name, articles_content)
        self.__save_file('%s_targets' % dataset_name, targets)

    def generate_dataset(self, train_questions, test_questions):
        logging.info('start generating dataset')
        try:
            self.__only_load_dataset()
        except:
            self.__load_data()
            self.__generate_dataset_with_questions(train_questions, 'train')
            self.__generate_dataset_with_questions(test_questions, 'test')
            self.__only_load_dataset()

    def __only_load_dataset(self):
        if self.__dataset_loaded:
            return

        self.__train_questions_id = self.__load_file('train_questions_id')
        self.__train_questions = self.__load_file('train_questions')
        self.__train_articles_id = self.__load_file('train_articles_id')
        self.__train_articles_title = self.__load_file('train_articles_title')
        self.__train_articles_content = self.__load_file('train_articles_content')
        self.__train_targets = self.__load_file('train_targets')

        self.__test_questions_id = self.__load_file('test_questions_id')
        self.__test_questions = self.__load_file('test_questions')
        self.__test_articles_id = self.__load_file('test_articles_id')
        self.__test_articles_title = self.__load_file('test_articles_title')
        self.__test_articles_content = self.__load_file('test_articles_content')
        self.__test_targets = self.__load_file('test_targets')

        logging.info("train samples: %d" % self.__train_questions.shape[0])
        logging.info("test samples: %d" % self.__test_questions.shape[0])

        logging.debug("train dataset:")
        logging.debug("questions_id: %s" % str(self.__train_questions_id.shape))
        logging.debug("questions: %s" % str(self.__train_questions.shape))
        logging.debug("articles_id: %s" % str(self.__train_articles_id.shape))
        logging.debug("articles_title: %s" % str(self.__train_articles_title.shape))
        logging.debug("articles_content: %s" % str(self.__train_articles_content.shape))
        logging.debug("targets: %s" % str(self.__train_targets.shape))

        logging.debug("train dataset:")
        logging.debug("questions_id: %s" % str(self.__test_questions_id.shape))
        logging.debug("questions: %s" % str(self.__test_questions.shape))
        logging.debug("articles_id: %s" % str(self.__test_articles_id.shape))
        logging.debug("articles_title: %s" % str(self.__test_articles_title.shape))
        logging.debug("articles_content: %s" % str(self.__test_articles_content.shape))
        logging.debug("targets: %s" % str(self.__test_targets.shape))

        self.__dataset_loaded = True

    def __simple_test_model(self, model, dataset_name, questions, articles_title, articles_content, target):
        test_scores = model.evaluate(
            { 'questions': questions, 'articles_title': articles_title, 'articles_content': articles_content },
            { 'weight': target },
            verbose = 0)
        logging.info('simple test, dataset: %s, loss: %.4f, accuracy: %.4f' % (dataset_name, test_scores[0], test_scores[1]))

    def __semi_test_model(self, model, dataset_name, questions, articles_title, articles_content, target):
        predictet_target = model.predict(
            { 'questions': questions, 'articles_title': articles_title, 'articles_content': articles_content },
            batch_size=64,
            verbose=0)
        predictet_target = predictet_target.reshape(-1)
        predictet_target = np.where(predictet_target > 0.5, 1.0, 0.0)
        corrected_count = np.count_nonzero(predictet_target == target)
        total_count = target.shape[0]
        r1 = self.__colored('%d/%d' % (corrected_count, total_count), 'yellow')
        r2 = self.__colored('%.2f %%' % (corrected_count / total_count * 100), 'yellow')
        logging.info("semi test, dataset: %s, corrected: %s (%s)" % (dataset_name, r1, r2))

    def _words_layers(self, filters, input):
        blocks = []
        input = tf.keras.layers.Conv1D(filters, 1, activation='relu')(input)
        for i in [2, 3]:
            block = tf.keras.layers.Conv1D(filters, i, activation='relu')(input)
            block = tf.keras.layers.GlobalMaxPooling1D()(block)
            blocks.append(block)
        block = tf.keras.layers.concatenate(blocks)
        return block
        block = tf.keras.layers.Reshape((4, filters))(block)
        block = tf.keras.layers.Conv1D(filters, 2, activation='relu')(block)
        block = tf.keras.layers.GlobalMaxPooling1D()(block)

    def __create_questions_model(self, filters):
        input = tf.keras.Input(shape=(self.__questions_words, NeuralWeightCalculator._W2V_SIZE), name='questions')
        block = self._words_layers(filters, input)
        return tf.keras.Model(input, block, name='questions_model')

    def __create_articles_model(self, filters):
        title_input = tf.keras.Input(shape=(self.__articles_title_words, NeuralWeightCalculator._W2V_SIZE), name='articles_title')
        title_block = self._words_layers(filters, title_input)

        content_input = tf.keras.Input(shape=(self.__articles_content_words, NeuralWeightCalculator._W2V_SIZE), name='articles_content')
        content_block = self._words_layers(filters, content_input)

        articles_block = tf.keras.layers.concatenate([title_block, content_block], name='articles_output')
        return tf.keras.Model([title_input, content_input], articles_block, name='articles_model')

    def _weight_layers(self, x):
        x = tf.keras.layers.Dense(32, activation='sigmoid')(x)
        x = tf.keras.layers.Dense(16, activation='sigmoid')(x)
        x = tf.keras.layers.Dense(1, activation='sigmoid', name='weight')(x)
        return x

    def __create_distances_model(self, filters, bypass_models):
        questions_model = self.__create_questions_model(filters)
        articles_model = self.__create_articles_model(filters)

        if not bypass_models:
            questions_input = tf.keras.Input(shape=(self.__questions_words, NeuralWeightCalculator._W2V_SIZE), name='questions')
            articles_title_input = tf.keras.Input(shape=(self.__articles_title_words, NeuralWeightCalculator._W2V_SIZE), name='articles_title')
            articles_content_input = tf.keras.Input(shape=(self.__articles_content_words, NeuralWeightCalculator._W2V_SIZE), name='articles_content')

            x = tf.keras.layers.concatenate([questions_model(questions_input), articles_model([articles_title_input, articles_content_input])])
            return tf.keras.Model(inputs=[questions_input, articles_title_input, articles_content_input], outputs=self._weight_layers(x), name='distances_model')
        else:
            questions_input = tf.keras.Input(shape=questions_model.output.shape[1:], name='questions')
            articles_input = tf.keras.Input(shape=articles_model.output.shape[1:], name='articles')
            x = tf.keras.layers.concatenate([questions_input, articles_input])
            return tf.keras.Model(inputs=[questions_input, articles_input], outputs=self._weight_layers(x), name='distances_model')

    def __create_model(self):
        model = self.__create_distances_model(NeuralWeightCalculator.__FILTERS, False)
        tf.keras.utils.plot_model(model, '%s/%s.png' % (self.__workdir, self._model_name()), show_shapes=True, expand_nested=True)
        return model

    def _model_name(self):
        return 'model'

    def __load_model(self):
        models_files = [f for f in os.listdir(self.__workdir) if re.match(r'%s_.*.h5' % self._model_name(), f)]
        best_model_file = sorted(models_files)[-1]
        logging.info('loading model from file: %s' % best_model_file)
        model = tf.keras.models.load_model('%s/%s' % (self.__workdir, best_model_file))
        return model

    def train(self, epoch):
        if epoch == 0:
            return

        logging.info('training model')

        try:
            model = self.__load_model()
            self.__simple_test_model(model, 'test', self.__test_questions, self.__test_articles_title, self.__test_articles_content, self.__test_targets)
            self.__semi_test_model(model, 'test', self.__test_questions, self.__test_articles_title, self.__test_articles_content, self.__test_targets)
        except:
            logging.info('create model')
            model = self.__create_model()

        logging.info('trainable weights: %s' % model.count_params())

        def on_epoch_end(current_epoch, data):
            try:
                logging.info("after epoch %d/%d, loss: %.6f, accuracy: %.6f, val_loss: %.6f, val_accuracy: %.6f" % (current_epoch, epoch, data['loss'], data['accuracy'], data['val_loss'], data['val_accuracy']))
                # self.__simple_test_model(model, 'test', test_questions, test_articles_title, test_articles_content, test_target)
                self.__semi_test_model(model, 'test', self.__test_questions, self.__test_articles_title, self.__test_articles_content, self.__test_targets)
            except:
                pass

        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath='%s/%s_{val_accuracy:.4f}.h5' % (self.__workdir, self._model_name()), save_best_only=True, monitor='val_accuracy', verbose=0)
        test_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

        model.compile(
            optimizer = tf.keras.optimizers.RMSprop(1e-3),
            loss = tf.keras.losses.BinaryCrossentropy(),
            loss_weights = [0.2],
            metrics = ['accuracy'])
        try:
            model.fit(
                { 'questions': self.__train_questions, 'articles_title': self.__train_articles_title, 'articles_content': self.__train_articles_content },
                { 'weight': self.__train_targets },
                batch_size = 64,
                epochs = epoch,
                validation_split = 0.2,
                verbose = 0,
                callbacks=[save_callback, test_callback])
            self.__simple_test_model(model, 'test', self.__test_questions, self.__test_articles_title, self.__test_articles_content, self.__test_targets)
            self.__semi_test_model(model, 'test', self.__test_questions, self.__test_articles_title, self.__test_articles_content, self.__test_targets)
        except KeyboardInterrupt:
            logging.info('learing stoppped by user')

    def __full_test_article(self, method, bypass_model, question_id, question_data, articles_id, articles_data):
        questions_data = np.repeat([question_data], articles_data.shape[0], 0)
        logging.info(questions_data.shape)
        logging.info(articles_data.shape)
        articles_weight = bypass_model.predict(
            { 'questions': questions_data, 'articles': articles_data},
            batch_size=256,
            verbose=0).reshape(-1)
        ResultsPresenter.present(Question.objects.get(id=question_id), articles_id, articles_weight, method, self.__debug_top_items, False)

    def __full_test_questions(self, method_name, questions_id, articles_id, articles_output, questions_model, bypass_model):
        test_method, created = Method.objects.get_or_create(name=method_name)
        questions_data = self.__prepare_questions(questions_id, self.__questions_words)
        logging.info('questions_data: %s' % str(questions_data.shape))
        questions_output = questions_model.predict(questions_data, batch_size=64, verbose=0)
        logging.debug('questions output: %s' % str(questions_output.shape))

        total = len(questions_id)
        for i in range(0, total):
            self.__full_test_article(test_method, bypass_model, questions_id[i], questions_output[i], articles_id, articles_output)
            logging.debug("questions progress: %d/%d (%.2f %%)" % (i, total, i / total * 100))

    def __full_test(self, method_name, test_questions_id, train_questions_id, articles_id, questions_articles_weight, model, questions_model, articles_model, bypass_model):
        self.__load_data()
        logging.info('full test')
        logging.info('test questions: %s' % str(test_questions_id.shape))
        logging.info('train questions: %s' % str(train_questions_id.shape))
        logging.info('articles: %s' % str(articles_id.shape))

        articles_output = np.zeros(shape=(0, articles_model.output_shape[1]))
        articles_id_chunks = np.array_split(articles_id, NeuralWeightCalculator.__ARTICLES_CHUNKS)
        current = 0
        total = NeuralWeightCalculator.__ARTICLES_CHUNKS
        for i in range(0, NeuralWeightCalculator.__ARTICLES_CHUNKS):
            articles_title = self.__prepare_articles(articles_id_chunks[i], self.__articles_title_words, True, False)
            articles_content = self.__prepare_articles(articles_id_chunks[i], self.__articles_content_words, False, False)
            chunk_output = articles_model.predict({ 'articles_title': articles_title, 'articles_content': articles_content }, batch_size=256, verbose=0)
            articles_output = np.concatenate((articles_output, chunk_output), axis=0)
            current += 1
            logging.debug("progress: %d/%d (%.2f %%)" % (current, total, current / total * 100))

        logging.debug('articles output: %s' % str(articles_output.shape))
        self.__full_test_questions('%s, dataset: test' % method_name, test_questions_id, articles_id, articles_output, questions_model, bypass_model)
        self.__full_test_questions('%s, dataset: train' % method_name, train_questions_id, articles_id, articles_output, questions_model, bypass_model)

    def __prepare_bypass_model(self, model, bypass_model):
        for l1 in model.layers:
            for l2 in bypass_model.layers:
                if type(l1) == type(l2) and l1.input_shape == l2.input_shape and l1.output_shape == l2.output_shape:
                    l2.set_weights(l1.get_weights())

    def __extract_models(self, model):
        questions_model = None
        articles_model = None
        for l in model.layers:
            if isinstance(l, tf.keras.Model):
                if l.name == "questions_model":
                    questions_model = l
                if l.name == "articles_model":
                    articles_model = l
        return (questions_model, articles_model)

    def __print_model(model):
        logging.debug(model.name)
        logging.debug("  inputs:")
        for i in model.inputs:
            logging.debug("    %s" % str(i.shape))
        logging.debug("  outputs:")
        for i in model.outputs:
            logging.debug("    %s" % str(i.shape))

    def test(self, method_name):
        logging.info('testing model')

        model = self.__load_model()
        (questions_model, articles_model) = self.__extract_models(model)
        bypass_model = self.__create_distances_model(NeuralWeightCalculator.__FILTERS, True)
        self.__prepare_bypass_model(model, bypass_model)
        NeuralWeightCalculator.__print_model(model)
        NeuralWeightCalculator.__print_model(questions_model)
        NeuralWeightCalculator.__print_model(articles_model)
        NeuralWeightCalculator.__print_model(bypass_model)
        questions_articles_weight = defaultdict(defaultdict)
        self.__simple_test_model(model, 'test', self.__test_questions, self.__test_articles_title, self.__test_articles_content, self.__test_targets)
        self.__semi_test_model(model, 'test', self.__test_questions, self.__test_articles_title, self.__test_articles_content, self.__test_targets)
        test_questions_id = np.unique(self.__test_questions_id)
        train_questions_id = np.unique(self.__train_questions_id)
        articles_id = np.array(Article.objects.values_list('id', flat=True))
        self.__full_test(method_name, test_questions_id, train_questions_id, articles_id, questions_articles_weight, model, questions_model, articles_model, bypass_model)
