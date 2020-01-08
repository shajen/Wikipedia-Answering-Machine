from collections import defaultdict, deque, Counter
from data.models import *
import logging
import math
import re
import numpy as np
import tools.results_presenter

class TfIdfWeightCalculator():
    def __init__(self, debug_top_items, ngram_size, data_loader):
        self.__debug_top_items = debug_top_items
        self.__ngram_size = ngram_size
        self.__data_loader = data_loader
        self.__articles_count = len(data_loader.get_articles_id())
        self.__questions_words_count = defaultdict(lambda: 0)
        logging.info('start parsing questions')
        for question_id in data_loader.get_questions_id():
            for ngram in TfIdfWeightCalculator.get_ngrams(data_loader.get_question_base_words(question_id), self.__ngram_size):
                self.__questions_words_count[ngram] += 1
        logging.info('finish parsing questions')

    def get_ngrams(words, ngram_size):
        ngrams = []
        for i in range(0, len(words) - ngram_size + 1):
            ngrams.append(tuple(words[i:i+ngram_size]))
        return ngrams

    def __count_articles(question_words, is_title, data_loader, ngram_size):
        logging.info("questions words: %d" % len(question_words))
        articles_words_count_ngram = defaultdict(lambda: defaultdict(lambda: 0))
        articles_positions_ngram = defaultdict(list)

        articles_words = data_loader.get_articles_base_words(is_title)
        articles_mask = np.isin(articles_words, question_words)

        for article_id in data_loader.get_articles_id():
            words = articles_words[article_id]
            mask = articles_mask[article_id]
            if np.any(mask):
                if ngram_size == 1:
                    indexes = np.arange(words.shape[0])[mask]
                    for index in indexes:
                        ngram = (words[index],)
                        articles_words_count_ngram[article_id][ngram] += 1
                        articles_positions_ngram[article_id].append((index, ngram))
                elif ngram_size == 2:
                    indexes = np.arange(words.shape[0])[mask]
                    for i in range(indexes.shape[0]-1):
                        if indexes[i] + 1 == indexes[i+1]:
                            ngram = (words[indexes[i]], words[indexes[i+1]])
                            articles_words_count_ngram[article_id][ngram] += 1
                            articles_positions_ngram[article_id].append((indexes[i], ngram))
        return (articles_words_count_ngram, articles_positions_ngram)

    def __count_idf(articles_count, articles_words_count):
        words_articles_count = defaultdict(lambda: 0)
        for item_id in articles_words_count:
            for word_id in articles_words_count[item_id]:
                words_articles_count[word_id] += 1

        articles_words_idf = {}
        questions_words_idf = {}
        for word_id in words_articles_count:
            articles_words_idf[word_id] = math.log(articles_count / words_articles_count[word_id])
            questions_words_idf[word_id] = articles_words_idf[word_id]

        # for word in set(word_to_representer.values()):
        #     questions_words_idf[word] = math.log(Question.objects.count() / questions_words_count[word])
        return (articles_words_idf, questions_words_idf)

    def __count_question_words_weights(question, words, questions_words_count, questions_words_idf):
        logging.debug('question words weights')
        question_words_weights = {}
        for (word, count) in Counter(words).items():
            try:
                tf = count / len(words)
                question_words_weights[word] = tf * questions_words_idf[word]
                if logging.getLogger().level == logging.DEBUG:
                    word_string = ', '.join(list(map(lambda x: str(Word.objects.get(id=x)), list(word))))
                    logging.debug(' - %-40s %d %.6f (%3d)' % (word_string, count, question_words_weights[word], questions_words_count[word]))
            except Exception as e:
                pass
                #logging.warning('exception during count question words weights')
                #logging.warning(e)
                #logging.warning('question: %s, word: %s' % (question, word))
        return question_words_weights

    def __print_debug_data(question, question_ngrams, articles_positions):
        if logging.getLogger().level != logging.DEBUG:
            return

        logging.info("XXX")
        logging.debug(question)
        logging.debug("words count: %d" % len(question_ngrams))
        for ngram in question_ngrams:
            logging.debug("ngram: %s" % str(ngram))
            for word_id in list(ngram):
                logging.debug("  word: %s" % (Word.objects.get(id=word_id).value))

        for answer in question.answer_set.select_related('article'):
            article = answer.article
            logging.debug('')
            logging.debug(article)
            logging.debug("words count: %d" % len(articles_positions[article.id]))
            for (pos, ngram) in articles_positions[article.id]:
                logging.debug("ngram: %s (pos: %d)" % (str(ngram), pos))
                for word_id in list(ngram):
                    logging.debug("  word: %s" % (Word.objects.get(id=word_id).value))

    def prepare(self, question, is_title):
        logging.info('preparing')
        question_words = self.__data_loader.get_question_base_words(question.id)
        question_ngrams = TfIdfWeightCalculator.get_ngrams(question_words, self.__ngram_size)
        (self.__articles_words_count, self.__articles_positions) = TfIdfWeightCalculator.__count_articles(question_words, is_title, self.__data_loader, self.__ngram_size)
        TfIdfWeightCalculator.__print_debug_data(question, question_ngrams, self.__articles_positions)
        (self.__articles_words_idf, questions_words_idf) = TfIdfWeightCalculator.__count_idf(self.__articles_count, self.__articles_words_count)
        self.__question_words_weights = TfIdfWeightCalculator.__count_question_words_weights(question, question_ngrams, self.__questions_words_count, questions_words_idf)

    def __dict_to_vector(keys, d):
        v = []
        for key in keys:
            try:
                v.append(d[key])
            except:
                v.append(0.0)
        return v

    def __convert_to_vector(question_words_weights, words_set_weights):
        keys = question_words_weights.keys()
        question_vector = TfIdfWeightCalculator.__dict_to_vector(keys, question_words_weights)
        vectors = []
        for weights in words_set_weights:
            vectors.append(TfIdfWeightCalculator.__dict_to_vector(keys, weights))
        return (question_vector, vectors)

    def __count_tf_idf(self, question, sum_neighbors, minimal_word_idf, comparators):
        logging.info('counting tf-idf')
        comparators_articles_words_weights = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        comparators_articles_weight = defaultdict(lambda: defaultdict())

        filtered_question_words_weights = dict(filter(lambda data: self.__articles_words_idf[data[0]] > minimal_word_idf, self.__question_words_weights.items()))
        for item_id in self.__articles_words_count:
            articles_words_set_weights = []
            # words_positions = filter(lambda data: self.__articles_words_idf[data[1]] > minimal_word_idf, self.__articles_positions[item_id])
            words_positions = self.__articles_positions[item_id]
            if sum_neighbors == 0:
                current_words = map(lambda data: data[1], words_positions)
                weights = {}
                for word_id, count in Counter(current_words).items():
                    weights[word_id] = count / len(self.__articles_positions[item_id]) * self.__articles_words_idf[word_id]
                articles_words_set_weights.append(weights)
            else:
                counter = defaultdict(lambda: 0)
                current_words = deque()
                if not words_positions:
                    continue
                for data in words_positions:
                    current_words.append(data)
                    counter[data[1]] += 1
                    while current_words[0][0] + sum_neighbors < current_words[-1][0]:
                        pop_data = current_words.popleft()
                        counter[pop_data[1]] -= 1
                        if counter[pop_data[1]] == 0:
                            del counter[pop_data[1]]

                    weights = {}
                    for word_id, count in counter.items():
                        weights[word_id] = count / sum_neighbors * self.__articles_words_idf[word_id]
                    articles_words_set_weights.append(weights)

            (question_vector, vectors) = TfIdfWeightCalculator.__convert_to_vector(filtered_question_words_weights, articles_words_set_weights)
            for comparator in comparators:
                try:
                    (best_weight, best_words_weights_index) = comparator.get_best_score(question_vector, vectors, articles_words_set_weights)
                    best_words_weights = articles_words_set_weights[best_words_weights_index]
                    for word_id in best_words_weights:
                        comparators_articles_words_weights[comparator.method()][item_id][word_id] = best_words_weights[word_id]
                    comparators_articles_weight[comparator.method()][item_id] = best_weight
                except ValueError as e:
                    logging.warn('exception during count articles weight')
                    logging.warn(e)
                    logging.warning('question: %s, article id: %s, fqww: %d, awsw: %d' % (question, item_id, len(filtered_question_words_weights), len(articles_words_set_weights)))

        return (comparators_articles_words_weights, comparators_articles_weight)

    def calculate(self, question, sum_neighbors, minimal_word_idf_weight, comparators):
        (comparators_articles_words_weight, comparators_articles_weight) = self.__count_tf_idf(question, sum_neighbors, minimal_word_idf_weight, comparators)
        for comparator in comparators:
            (articles_words_weight, articles_weight) = (comparators_articles_words_weight[comparator.method()], comparators_articles_weight[comparator.method()])
            articles_id = list(articles_weight.keys())
            distances = np.array(list(articles_weight.values()))
            (method, created) = Method.objects.get_or_create(name=comparator.method())
            tools.results_presenter.ResultsPresenter.present(question, articles_id, distances, method, self.__debug_top_items, not comparator.ascending_order())
