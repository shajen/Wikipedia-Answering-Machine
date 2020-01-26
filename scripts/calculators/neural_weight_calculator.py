from data.models import *
from functools import reduce
from termcolor import colored
from tools.results_presenter import ResultsPresenter
import logging
import numpy as np
import os
import tensorflow as tf
import re
import gc

class NeuralWeightCalculator():
    __ARTICLES_PER_CHUNK = 1000
    __FILTERS = 64
    __PREDICT_BATCH_SIZE = 20000
    __TRAIN_BATCH_SIZE = 64

    def __init__(self, data_loader, debug_top_items, workdir, good_bad_ratio, method_id):
        self.__data_loader = data_loader
        self.__debug_top_items = debug_top_items
        self.__workdir = workdir
        self._word2vec_size = data_loader.word2vec_size()
        self.__questions_words = data_loader.questions_words_count()
        self.__articles_title_words = data_loader.articles_title_words_count()
        self.__articles_content_words = data_loader.articles_content_words_count()
        self.__good_bad_ratio = good_bad_ratio
        self.__dataset_loaded = False
        self.__method_id = method_id

    def __colored(self, text, colour):
        return colored(text, colour, attrs={'bold'})

    def __prepare_articles(self, articles_id, top_words, is_title, show_progress):
        logging.debug('preparing %d articles, top words: %d, title: %d' % (articles_id.shape[0], top_words, is_title))
        if is_title:
            data = np.zeros(shape=(articles_id.shape[0], self.__articles_title_words, self._word2vec_size), dtype=np.float32)
        else:
            data = np.zeros(shape=(articles_id.shape[0], self.__articles_content_words, self._word2vec_size), dtype=np.float32)
        total = articles_id.shape[0]
        current = 0
        step = max(1, round(total / (10 if is_title else 100)))
        for article_id in articles_id:
            words = self.__data_loader.get_article_words_id(False, article_id, is_title)
            data[current] = np.nan_to_num(self.__data_loader.get_words_data(words))
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
        step = max(1, round(total / 10))
        for question_id in questions_id:
            words = self.__data_loader.get_question_words_id(False, question_id)
            data.append(np.nan_to_num(self.__data_loader.get_words_data(words)))
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

    def __get_bad_articles_id(self, questions_id, good_articles_id):
        if self.__method_id == 0:
            return list(Article.objects.exclude(id__in=good_articles_id).order_by('?').values_list('id', flat=True)[:self.__good_bad_ratio * len(good_articles_id)])
        else:
            is_smaller_first = Method.objects.get(id=self.__method_id).is_smaller_first
            articles_id = []
            for question_id in questions_id:
                corrected_articles_id = Answer.objects.filter(question_id=question_id).values_list('article_id', flat=True)
                count = self.__good_bad_ratio * len(corrected_articles_id)
                order_by = 'weight' if is_smaller_first else '-weight'
                articles_id.extend(Rate.objects.filter(method_id=self.__method_id, question_id=question_id).exclude(article_id__in=corrected_articles_id).order_by(order_by)[:count].values_list('article_id', flat=True))
            return articles_id

    def __generate_dataset_with_questions(self, questions, dataset_name):
        logging.info('generating dataset: %s' % dataset_name)

        logging.debug('questions: %d' % len(questions))
        questions_id = list(map(lambda q: [q.id] * q.answer_set.count(), questions))
        questions_id = list(reduce(lambda x, y: x + y, questions_id, []))
        logging.debug('answers: %d' % len(questions_id))

        good_articles_id = list(map(lambda q: list(map(lambda a: a.article_id, q.answer_set.all())), questions))
        good_articles_id = list(reduce(lambda x, y: x + y, good_articles_id, []))
        logging.debug('good articles: %d' % len(good_articles_id))

        bad_articles_id = self.__get_bad_articles_id(list(map(lambda q: q.id, questions)), good_articles_id)
        logging.debug('bad articles: %d' % len(bad_articles_id))

        articles_id = np.concatenate((np.array(good_articles_id), np.array(bad_articles_id)))
        logging.debug('total articles: %d' % len(articles_id))

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

        self.__save_file('nn_%s_questions_id' % dataset_name, questions_id)
        self.__save_file('nn_%s_questions' % dataset_name, questions)
        self.__save_file('nn_%s_articles_id' % dataset_name, articles_id)
        self.__save_file('nn_%s_articles_title' % dataset_name, articles_title)
        self.__save_file('nn_%s_articles_content' % dataset_name, articles_content)
        self.__save_file('nn_%s_targets' % dataset_name, targets)

    def __generate_dataset(self, train_questions, validate_questions, test_questions):
        logging.info('start generating dataset')
        try:
            self.__only_load_all_dataset()
        except:
            self.__generate_dataset_with_questions(train_questions, 'train')
            self.__generate_dataset_with_questions(validate_questions, 'validate')
            self.__generate_dataset_with_questions(test_questions, 'test')
            self.__only_load_all_dataset()

    def __only_load_dataset(self, dataset):
        questions_id = self.__load_file('nn_%s_questions_id' % dataset)
        questions = self.__load_file('nn_%s_questions' % dataset)
        articles_id = self.__load_file('nn_%s_articles_id' % dataset)
        articles_title = self.__load_file('nn_%s_articles_title' % dataset)
        articles_content = self.__load_file('nn_%s_articles_content' % dataset)
        targets = self.__load_file('nn_%s_targets' % dataset)

        logging.debug("%s dataset:" % dataset)
        logging.debug("samples: %d" % (questions.shape[0]))
        logging.debug("questions_id: %s" % str(questions_id.shape))
        logging.debug("questions: %s" % str(questions.shape))
        logging.debug("articles_id: %s" % str(articles_id.shape))
        logging.debug("articles_title: %s" % str(articles_title.shape))
        logging.debug("articles_content: %s" % str(articles_content.shape))
        logging.debug("targets: %s" % str(targets.shape))
        return (questions_id, questions, articles_id, articles_title, articles_content, targets)

    def __only_load_all_dataset(self):
        if self.__dataset_loaded:
            return
        (self.__train_questions_id, self.__train_questions, self.__train_articles_id, self.__train_articles_title, self.__train_articles_content, self.__train_targets) = self.__only_load_dataset('train')
        (self.__validate_questions_id, self.__validate_questions, self.__validate_articles_id, self.__validate_articles_title, self.__validate_articles_content, self.__validate_targets) = self.__only_load_dataset('validate')
        (self.__test_questions_id, self.__test_questions, self.__test_articles_id, self.__test_articles_title, self.__test_articles_content, self.__test_targets) = self.__only_load_dataset('test')
        self.__dataset_loaded = True

    def __simple_test_model(self, model, dataset_name, questions, articles_title, articles_content, target):
        test_scores = model.evaluate(
            { 'questions': questions, 'articles_title': articles_title, 'articles_content': articles_content },
            { 'weight': target },
            verbose = 0)
        logging.info('simple test, dataset: %s, loss: %.4f, accuracy: %.4f' % (dataset_name.rjust(10), test_scores[0], test_scores[1]))

    def __semi_test_model(self, model, dataset_name, questions, articles_title, articles_content, target):
        predictet_target = model.predict(
            { 'questions': questions, 'articles_title': articles_title, 'articles_content': articles_content },
            batch_size=NeuralWeightCalculator.__PREDICT_BATCH_SIZE,
            verbose=0)
        predictet_target = predictet_target.reshape(-1)
        predictet_target = np.where(predictet_target > 0.5, 1.0, 0.0)
        corrected_count = np.count_nonzero(predictet_target == target)
        total_count = target.shape[0]
        r1 = self.__colored('%d/%d' % (corrected_count, total_count), 'yellow')
        r2 = self.__colored('%.2f %%' % (corrected_count / total_count * 100), 'yellow')
        logging.info("semi test, dataset: %s, corrected: %s (%s)" % (dataset_name.rjust(10), r1, r2))

    def __test_model(self, dataset, model):
        if dataset == 'train':
            self.__simple_test_model(model, dataset, self.__train_questions, self.__train_articles_title, self.__train_articles_content, self.__train_targets)
            # self.__semi_test_model(model, dataset, self.__train_questions, self.__train_articles_title, self.__train_articles_content, self.__train_targets)
        elif dataset == 'validate':
            self.__simple_test_model(model, dataset, self.__validate_questions, self.__validate_articles_title, self.__validate_articles_content, self.__validate_targets)
            # self.__semi_test_model(model, dataset, self.__validate_questions, self.__validate_articles_title, self.__validate_articles_content, self.__validate_targets)
        elif dataset == 'test':
            self.__simple_test_model(model, dataset, self.__test_questions, self.__test_articles_title, self.__test_articles_content, self.__test_targets)
            # self.__semi_test_model(model, dataset, self.__test_questions, self.__test_articles_title, self.__test_articles_content, self.__test_targets)

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
        input = tf.keras.Input(shape=(self.__questions_words, self._word2vec_size), name='questions')
        block = self._words_layers(filters, input)
        return tf.keras.Model(input, block, name='questions_model')

    def __create_articles_model(self, filters):
        title_input = tf.keras.Input(shape=(self.__articles_title_words, self._word2vec_size), name='articles_title')
        title_block = self._words_layers(filters, title_input)

        content_input = tf.keras.Input(shape=(self.__articles_content_words, self._word2vec_size), name='articles_content')
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
            questions_input = tf.keras.Input(shape=(self.__questions_words, self._word2vec_size), name='questions')
            articles_title_input = tf.keras.Input(shape=(self.__articles_title_words, self._word2vec_size), name='articles_title')
            articles_content_input = tf.keras.Input(shape=(self.__articles_content_words, self._word2vec_size), name='articles_content')

            x = tf.keras.layers.concatenate([questions_model(questions_input), articles_model([articles_title_input, articles_content_input])])
            return tf.keras.Model(inputs=[questions_input, articles_title_input, articles_content_input], outputs=self._weight_layers(x), name='distances_model')
        else:
            questions_input = tf.keras.Input(shape=questions_model.output.shape[1:], name='questions')
            articles_input = tf.keras.Input(shape=articles_model.output.shape[1:], name='articles')
            x = tf.keras.layers.concatenate([questions_input, articles_input])
            return tf.keras.Model(inputs=[questions_input, articles_input], outputs=self._weight_layers(x), name='distances_model')

    def __create_model(self):
        model = self.__create_distances_model(NeuralWeightCalculator.__FILTERS, False)
        tf.keras.utils.plot_model(model, '%s/%s.png' % (self.__workdir, self.model_name()), show_shapes=True, expand_nested=True)
        return model

    def model_name(self):
        return 'cnn_model'

    def __load_model(self):
        models_files = [f for f in os.listdir(self.__workdir) if re.match(r'%s_.*.h5' % self.model_name(), f)]
        best_model_file = sorted(models_files)[-1]
        logging.info('loading model from file: %s' % best_model_file)
        model = tf.keras.models.load_model('%s/%s' % (self.__workdir, best_model_file))
        return model

    def train(self, train_questions, validate_questions, test_questions, epoch):
        self.__generate_dataset(train_questions, validate_questions, test_questions)

        if epoch == 0:
            return

        logging.info('training: %s' % self.model_name())

        try:
            model = self.__load_model()
            self.__test_model('train', model)
            self.__test_model('validate', model)
        except:
            logging.info('create model')
            model = self.__create_model()

        logging.info('trainable weights: %s' % model.count_params())

        def on_epoch_end(current_epoch, data):
            try:
                logging.info("after epoch %d/%d, loss: %.6f, accuracy: %.6f, val_loss: %.6f, val_accuracy: %.6f" % (current_epoch+1, epoch, data['loss'], data['accuracy'], data['val_loss'], data['val_accuracy']))
                # self.__test_model('train', model)
                # self.__test_model('validate', model)
            except:
                pass

        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath='%s/%s_{val_accuracy:.4f}.h5' % (self.__workdir, self.model_name()), save_best_only=True, monitor='val_accuracy', verbose=0)
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
                batch_size = NeuralWeightCalculator.__TRAIN_BATCH_SIZE,
                epochs = epoch,
                validation_data = (
                    { 'questions': self.__validate_questions, 'articles_title': self.__validate_articles_title, 'articles_content': self.__validate_articles_content },
                    { 'weight': self.__validate_targets }
                ),
                verbose = 0,
                callbacks=[save_callback, test_callback])
        except KeyboardInterrupt:
            logging.info('learing stoppped by user')

        gc.collect()
        self.__test_model('train', model)
        self.__test_model('validate', model)
        self.__test_model('test', model)

    def __full_test_article(self, method, bypass_model, question_id, question_data, articles_id, articles_data):
        questions_data = np.repeat([question_data], articles_data.shape[0], 0)
        articles_weight = bypass_model.predict(
            { 'questions': questions_data, 'articles': articles_data},
            batch_size=NeuralWeightCalculator.__PREDICT_BATCH_SIZE,
            verbose=0).reshape(-1)
        ResultsPresenter.present(Question.objects.get(id=question_id), list(articles_id), articles_weight, method, self.__debug_top_items, False)

    def __full_test_questions(self, method_name, question_id, articles_id, articles_output, questions_model, bypass_model):
        test_method, created = Method.objects.get_or_create(name=method_name, is_smaller_first=False)
        questions_data = self.__prepare_questions(np.array([question_id]), self.__questions_words)
        questions_output = questions_model.predict(questions_data, batch_size=NeuralWeightCalculator.__PREDICT_BATCH_SIZE, verbose=0)
        self.__full_test_article(test_method, bypass_model, question_id, questions_output[0], articles_id, articles_output)

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

    def __prepare_articles_for_testing(self, model):
        gc.collect()
        NeuralWeightCalculator.__print_model(model)
        NeuralWeightCalculator.__print_model(self.__questions_model)
        NeuralWeightCalculator.__print_model(self.__articles_model)
        NeuralWeightCalculator.__print_model(self.__bypass_model)
        self.__test_model('train', model)
        self.__test_model('validate', model)
        self.__test_model('test', model)

        self.__articles_id = self.__data_loader.get_articles_id()
        self.__articles_output = np.zeros(shape=(0, self.__articles_model.output_shape[1]))
        chunks = int(self.__articles_id.shape[0] / NeuralWeightCalculator.__ARTICLES_PER_CHUNK)
        articles_id_chunks = np.array_split(self.__articles_id, chunks)
        current = 0
        for i in range(0, chunks):
            gc.collect()
            articles_title = self.__prepare_articles(articles_id_chunks[i], self.__articles_title_words, True, False)
            articles_content = self.__prepare_articles(articles_id_chunks[i], self.__articles_content_words, False, False)
            chunk_output = self.__articles_model.predict({ 'articles_title': articles_title, 'articles_content': articles_content }, batch_size=NeuralWeightCalculator.__PREDICT_BATCH_SIZE, verbose=0)
            self.__articles_output = np.concatenate((self.__articles_output, chunk_output), axis=0)
            current += 1
            logging.info("progress: %d/%d (%.2f %%)" % (current, chunks, current / chunks * 100))

    def prepare_for_testing(self):
        logging.info('prepare for testing: %s' % self.model_name())

        model = self.__load_model()
        (self.__questions_model, self.__articles_model) = self.__extract_models(model)
        self.__bypass_model = self.__create_distances_model(NeuralWeightCalculator.__FILTERS, True)
        self.__prepare_bypass_model(model, self.__bypass_model)

        try:
            self.__articles_id = self.__load_file(self.model_name() + '_articles_id')
            self.__articles_output = self.__load_file(self.model_name() + '_articles_output')
        except FileNotFoundError:
            self.__prepare_articles_for_testing(model)
            self.__save_file(self.model_name() + '_articles_id', self.__articles_id)
            self.__save_file(self.model_name() + '_articles_output', self.__articles_output)

        logging.info('articles id: %s' % str(self.__articles_id.shape))
        logging.info('articles output: %s' % str(self.__articles_output.shape))
        gc.collect()
        self.__articles_output = tf.convert_to_tensor(self.__articles_output)

    def test(self, question, method_name):
        gc.collect()
        self.__full_test_questions(method_name, question.id, self.__articles_id, self.__articles_output, self.__questions_model, self.__bypass_model)
