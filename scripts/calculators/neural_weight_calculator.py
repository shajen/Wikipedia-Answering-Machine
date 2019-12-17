from collections import defaultdict
from data.models import *
from functools import reduce
from termcolor import colored
import gensim.models
import logging
import numpy as np
import os
import sys
import tensorflow as tf

class NeuralWeightCalculator():
    __W2V_SIZE = 100
    __ARTICLES_CHUNKS = 10000

    def __init__(self, debug_top_items, model_file, workdir):
        self.__debug_top_items = debug_top_items
        self.__model_file = model_file
        self.__workdir = workdir
        self.__data_loaded = False

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
            return np.array([0.0] * NeuralWeightCalculator.__W2V_SIZE)

    def __words2vec(self, words):
        return list(map(lambda word: self.__word2vec(word), words))

    def __prepare_article(self, article_id, words, is_title):
        article_words = [None] * words
        for (word_id, positions) in ArticleOccurrence.objects.filter(article_id=article_id, is_title=is_title).values_list('word_id', 'positions').iterator():
            for position in positions.split(','):
                try:
                    position = int(position)
                    if (position <= words):
                        article_words[position - 1] = word_id
                except:
                    pass
        return self.__words2vec(article_words)

    def __prepare_articles(self, articles_id, words, name, is_title, show_progress):
        logging.debug('preparing %d articles, top words: %d, title: %d' %(len(articles_id), words, is_title))
        data = []
        total = len(articles_id)
        current = 1
        step = round(total / (10 if is_title else 100))
        for article_id in articles_id:
            data.append(self.__prepare_article(article_id, words, is_title))
            current += 1
            if current % step == 0 and show_progress:
                logging.debug("progress: %d/%d (%.2f %%)" % (current, total, current / total * 100))
        self.__save_file(name, np.array(data))

    def __prepare_question(self, question_id, words):
        question_words = [None] * words
        for (word_id, positions) in QuestionOccurrence.objects.filter(question_id=question_id).values_list('word_id', 'positions').iterator():
            for position in positions.split(','):
                position = int(position)
                if (position <= words):
                    question_words[position - 1] = word_id
        return self.__words2vec(question_words)

    def __prepare_questions(self, questions_id, words):
        logging.debug('preparing %d questions, top words: %d' %(len(questions_id), words))
        data = []
        total = len(questions_id)
        current = 1
        step = round(total / 10)
        for question_id in questions_id:
            data.append(self.__prepare_question(question_id, words))
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

    def prepare_data(self, questions, questionWords, articleTitleWords, articleContentWords, goodBadArticlesRatio):
        logging.info('start preparing data')
        self.__load_data()

        questions_id = list(map(lambda q: [q.id] * q.answer_set.count(), questions))
        questions_id = list(reduce(lambda x, y: x + y, questions_id, []))
        logging.debug('questions: %d' % len(questions_id))

        good_articles_id = list(map(lambda q: list(map(lambda a: a.article_id, q.answer_set.all())), questions))
        good_articles_id = list(reduce(lambda x, y: x + y, good_articles_id, []))
        logging.debug('good articles: %d' % len(good_articles_id))

        bad_articles_id = list(Article.objects.exclude(id__in=good_articles_id).order_by('?').values_list('id', flat=True)[:goodBadArticlesRatio * len(good_articles_id)])
        logging.debug('bad articles: %d' % len(bad_articles_id))

        self.__prepare_articles(good_articles_id, articleTitleWords, 'good_articles_title_data', True, True)
        self.__prepare_articles(good_articles_id, articleContentWords, 'good_articles_content_data', False, True)
        self.__prepare_articles(bad_articles_id, articleTitleWords, 'bad_articles_title_data', True, True)
        self.__prepare_articles(bad_articles_id, articleContentWords, 'bad_articles_content_data', False, True)
        good_questions_data = self.__prepare_questions(questions_id, questionWords)
        good_target = np.array([1.0] * good_questions_data.shape[0])
        bad_questions_data = np.random.permutation(np.repeat(good_questions_data, goodBadArticlesRatio, axis=0))
        bad_target = np.array([0.0] * bad_questions_data.shape[0])

        self.__save_file('good_questions_data', good_questions_data)
        self.__save_file('good_target', good_target)
        self.__save_file('bad_questions_data', bad_questions_data)
        self.__save_file('bad_target', bad_target)

    def __prepare_dataset(self, train_data_percentage):
        logging.info('preparing dataset')
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
        train_questions = questions[:split_index]
        train_articles_title = articles_title[:split_index]
        train_articles_content = articles_content[:split_index]
        train_target = target[:split_index]

        test_questions = questions[split_index:]
        test_articles_title = articles_title[split_index:]
        test_articles_content = articles_content[split_index:]
        test_target = target[split_index:]

        logging.info("train samples: %d" %train_questions.shape[0])
        logging.info("test samples: %d" %test_questions.shape[0])

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

        return ((train_questions, train_articles_title, train_articles_content, train_target),
                (test_questions, test_articles_title, test_articles_content, test_target))

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

    def __create_questions_model(self, filters, questions_size):
        input = tf.keras.Input(shape=(questions_size, NeuralWeightCalculator.__W2V_SIZE), name='questions')
        block = tf.keras.layers.Conv1D(filters, 5, activation='relu')(input)
        block = tf.keras.layers.AveragePooling1D(2)(block)
        block = tf.keras.layers.Conv1D(filters, 4, activation='relu', name='questions_output')(block)
        return tf.keras.Model(input, block, name='questions_model')

    def __create_articles_model(self, filters, title_size, content_size):
        title_input = tf.keras.Input(shape=(title_size, NeuralWeightCalculator.__W2V_SIZE), name='articles_title')
        title_block = tf.keras.layers.Conv1D(filters, 5, activation='relu')(title_input)
        title_block = tf.keras.layers.AveragePooling1D(2)(title_block)
        title_block = tf.keras.layers.Conv1D(filters, 4, activation='relu')(title_block)

        content_input = tf.keras.Input(shape=(content_size, NeuralWeightCalculator.__W2V_SIZE), name='articles_content')
        content_block = tf.keras.layers.Conv1D(filters, 5, activation='relu')(content_input)
        content_block = tf.keras.layers.AveragePooling1D(2)(content_block)
        content_block = tf.keras.layers.Conv1D(filters, 5, activation='relu')(content_block)
        content_block = tf.keras.layers.AveragePooling1D(2)(content_block)
        content_block = tf.keras.layers.Conv1D(filters, 5, activation='relu')(content_block)
        content_block = tf.keras.layers.AveragePooling1D(2)(content_block)
        content_block = tf.keras.layers.Conv1D(filters, 5, activation='relu')(content_block)

        articles_block = tf.keras.layers.add([title_block, content_block], name='articles_output')
        return tf.keras.Model([title_input, content_input], articles_block, name='articles_model')

    def __create_distances_model(self, filters, questions_size, articles_title_size, articles_content_size, bypass_models):
        questions_model = self.__create_questions_model(filters, questions_size)
        articles_model = self.__create_articles_model(filters, articles_title_size, articles_content_size)

        def add_layers(x):
            x = tf.keras.layers.Conv1D(filters, 2, activation='relu')(x)
            x = tf.keras.layers.Conv1D(filters, 2, activation='relu')(x)
            x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(20, activation='sigmoid')(x)
            x = tf.keras.layers.Dense(1, activation='sigmoid', name='weight')(x)
            return x

        if not bypass_models:
            questions_input = tf.keras.Input(shape=(questions_size, NeuralWeightCalculator.__W2V_SIZE), name='questions')
            articles_title_input = tf.keras.Input(shape=(articles_title_size, NeuralWeightCalculator.__W2V_SIZE), name='articles_title')
            articles_content_input = tf.keras.Input(shape=(articles_content_size, NeuralWeightCalculator.__W2V_SIZE), name='articles_content')

            x = tf.keras.layers.add([questions_model(questions_input), articles_model([articles_title_input, articles_content_input])])
            return tf.keras.Model(inputs=[questions_input, articles_title_input, articles_content_input], outputs=add_layers(x), name='distances_model')
        else:
            questions_input = tf.keras.Input(shape=questions_model.output.shape[1:], name='questions')
            articles_input = tf.keras.Input(shape=articles_model.output.shape[1:], name='articles')
            x = tf.keras.layers.add([questions_input, articles_input])
            return tf.keras.Model(inputs=[questions_input, articles_input], outputs=add_layers(x), name='distances_model')

    def __create_model(self, questions_size, articles_title_size, articles_content_size):
        model = self.__create_distances_model(20, questions_size, articles_title_size, articles_content_size, False)
        tf.keras.utils.plot_model(model, '%s/model.png' % self.__workdir, show_shapes=True, expand_nested=True)
        return model

    def train(self, use_last_trained, epoch, train_data_percentage):
        logging.info('training model')
        (train_data, test_data) = self.__prepare_dataset(train_data_percentage)
        (train_questions, train_articles_title, train_articles_content, train_target) = train_data
        (test_questions, test_articles_title, test_articles_content, test_target) = test_data

        if use_last_trained:
            logging.info('use last trained model')
            model = tf.keras.models.load_model('%s/model.h5' % self.__workdir)
            self.__simple_test_model(model, 'test', test_questions, test_articles_title, test_articles_content, test_target)
            self.__semi_test_model(model, 'test', test_questions, test_articles_title, test_articles_content, test_target)

        else:
            logging.info('create model')
            model = self.__create_model(train_questions.shape[1], train_articles_title.shape[1], train_articles_content.shape[1])
        logging.info('trainable weights: %s' % model.count_params())

        def on_epoch_end(current_epoch, data):
            try:
                logging.info("after epoch %d/%d, loss: %.6f, accuracy: %.6f, val_loss: %.6f, val_accuracy: %.6f" % (current_epoch, epoch, data['loss'], data['accuracy'], data['val_loss'], data['val_accuracy']))
                # self.__simple_test_model(model, 'test', test_questions, test_articles_title, test_articles_content, test_target)
                self.__semi_test_model(model, 'test', test_questions, test_articles_title, test_articles_content, test_target)
            except:
                pass

        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath='%s/model.h5' % self.__workdir, save_best_only=True, monitor='val_accuracy', verbose=0)
        test_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)

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
                verbose = 0,
                callbacks=[save_callback, test_callback])
            self.__simple_test_model(model, 'test', test_questions, test_articles_title, test_articles_content, test_target)
            self.__semi_test_model(model, 'test', test_questions, test_articles_title, test_articles_content, test_target)
        except KeyboardInterrupt:
            logging.info('learing stoppped by user')

    def  __prepare_all_questions(self, questions_words):
        logging.info('preparing questions')
        questions_id = np.array(list(Question.objects.all().values_list('id', flat=True)))
        if os.path.isfile('%s/questions_id.npy' % self.__workdir) and os.path.isfile('%s/questions_data.npy' % self.__workdir):
            saved_data_ok = np.array_equal(self.__load_file('questions_id'), questions_id)
        else:
            saved_data_ok = False
        logging.info('saved data ok: %d' % saved_data_ok)
        if not saved_data_ok:
            self.__load_data()
            questions_data = self.__prepare_questions(questions_id, questions_words)
            self.__save_file('questions_id', questions_id)
            self.__save_file('questions_data', questions_data)

    def  __prepare_all_articles(self, articles_title_words, articles_content_words):
        logging.info('preparing articles chunks')
        articles_id = np.array(list(Article.objects.order_by('id').values_list('id', flat=True)))
        if os.path.isfile('%s/articles/articles_id.npy' % self.__workdir):
            saved_data_ok = np.array_equal(self.__load_file('articles/articles_id'), articles_id)
        else:
            saved_data_ok = False
        logging.info('saved data ok: %d' % saved_data_ok)

        if not saved_data_ok:
            self.__save_file('articles/articles_id', articles_id)
        chunks = np.array_split(articles_id, NeuralWeightCalculator.__ARTICLES_CHUNKS)
        i = 1
        total = len(chunks)
        for chunk_articles_id in chunks:
            if not os.path.isfile('%s/articles/articles_content_chunk_%03d.npy' % (self.__workdir, i)) or not saved_data_ok:
                self.__load_data()
                self.__prepare_articles(chunk_articles_id, articles_title_words, 'articles/articles_title_chunk_%03d' % i, True, False)
                self.__prepare_articles(chunk_articles_id, articles_content_words, 'articles/articles_content_chunk_%03d' % i, False, False)
                logging.debug("progress: %d/%d (%.2f %%)" % (i, total, i / total * 100))
            # else:
            #     logging.debug("chunk %d already exists, skipping" % i)
            i += 1

    def __full_test_articles(self, articles_model, bypass_model, questions_id, questions_output, articles_id, articles_title, articles_content):
        logging.debug("questions: %s, articles id: %s, title: %s, content: %s" % (str(questions_output.shape), str(articles_id.shape), str(articles_title.shape), str(articles_content.shape)))
        articles_output = articles_model.predict(
            { 'articles_title': articles_title, 'articles_content': articles_content },
            batch_size=256,
            verbose=0)
        articles_output = np.repeat(articles_output, questions_id.shape[0], 0)
        questions_output = np.repeat([questions_output], articles_id.shape[0], 0).reshape(articles_id.shape[0] * questions_output.shape[0], questions_output.shape[1], questions_output.shape[2])
        predictet_target = bypass_model.predict(
            { 'questions': questions_output, 'articles': articles_output},
            batch_size=256,
            verbose=0)

    def __full_test(self, model, questions_model, articles_model, bypass_model):
        logging.info('full test')
        questions_id = self.__load_file('questions_id')
        questions_data = self.__load_file('questions_data')
        questions_output = predictet_target = questions_model.predict(questions_data, batch_size=64, verbose=0)
        articles_id = self.__load_file('articles/articles_id')
        article_counter = 0
        total = NeuralWeightCalculator.__ARTICLES_CHUNKS
        for i in range(1, NeuralWeightCalculator.__ARTICLES_CHUNKS + 1):
            chunks_articles_title_data = self.__load_file('articles/articles_title_chunk_%03d' % i)
            chunks_articles_content_data = self.__load_file('articles/articles_content_chunk_%03d' % i)
            chunks_articles_id = articles_id[article_counter:article_counter+chunks_articles_title_data.shape[0]]
            article_counter += chunks_articles_title_data.shape[0]
            self.__full_test_articles(articles_model, bypass_model, questions_id, questions_output, chunks_articles_id, chunks_articles_title_data, chunks_articles_content_data)
            logging.debug("progress: %d/%d (%.2f %%)" % (i, total, i / total * 100))

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

    def test(self, train_data_percentage):
        logging.info('testing model')
        (train_data, test_data) = self.__prepare_dataset(train_data_percentage)
        (test_questions, test_articles_title, test_articles_content, test_target) = test_data

        model = tf.keras.models.load_model('%s/model.h5' % self.__workdir)
        (questions_model, articles_model) = self.__extract_models(model)
        bypass_model = self.__create_distances_model(20, test_questions.shape[1], test_articles_title.shape[1], test_articles_content.shape[1], True)
        self.__prepare_bypass_model(model, bypass_model)
        NeuralWeightCalculator.__print_model(model)
        NeuralWeightCalculator.__print_model(questions_model)
        NeuralWeightCalculator.__print_model(articles_model)
        NeuralWeightCalculator.__print_model(bypass_model)
        self.__simple_test_model(model, 'test', test_questions, test_articles_title, test_articles_content, test_target)
        self.__semi_test_model(model, 'test', test_questions, test_articles_title, test_articles_content, test_target)
        self.__prepare_all_questions(test_questions.shape[1])
        self.__prepare_all_articles(test_articles_title.shape[1], test_articles_content.shape[1])
        self.__full_test(model, questions_model, articles_model, bypass_model)
