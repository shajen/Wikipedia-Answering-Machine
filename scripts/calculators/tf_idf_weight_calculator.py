from collections import defaultdict, deque, Counter
from data.models import *
import logging
import math
import re
import numpy as np
import tools.results_presenter

class TfIdfWeightCalculator():
    def __init__(self, debug_top_items, ngram):
        self.debug_top_items = debug_top_items
        self.ngram = ngram
        logging.info('start reading articles')
        self.articles_count = Article.objects.filter(content_words_count__gte=20).count()
        logging.info('finish reading')
        logging.info('start parsing questions')
        self.questions_words_count = defaultdict(lambda: 0)
        for question in Question.objects.all():
            for word in question.get_ngrams(ngram):
                self.questions_words_count[word] += 1
        logging.info('finish parsing questions')

    def __get_words(question, debug=False):
        word_to_representer = {}
        for occurrence in QuestionOccurrence.objects.filter(question=question):
            org_word = occurrence.word
            if org_word.is_stop_word:
                continue
            for base_word in list(WordForm.objects.filter(changed_word=org_word.id).values_list('base_word', flat=True)) + [org_word.id]:
                word_to_representer[base_word] = org_word.id
                for changed_word in WordForm.objects.filter(base_word_id=base_word).values_list('changed_word', flat=True):
                    word_to_representer[changed_word] = org_word.id

        if debug:
            for representer in set(word_to_representer.values()):
                words = filter(lambda data: data[1] == representer, word_to_representer.items())
                words = list(map(lambda data: data[0], words))
                words = Word.objects.filter(id__in=words).values_list('value', flat=True)
                words = ', '.join(words)
                logging.debug('%s (%s)' % (Word.objects.get(id=representer), words))
        return word_to_representer

    def __count_articles(words, word_to_representer, is_title, ngram):
        articles_words_count = defaultdict(lambda: defaultdict(lambda: 0))
        # articles_words_positions = defaultdict(defaultdict)
        articles_positions = defaultdict(list)
        occurrences = ArticleOccurrence.objects.filter(word_id__in=word_to_representer.keys(), is_title=is_title).values('article_id', 'word_id', 'positions_count', 'positions')
        for occurrence in occurrences:
            representer = word_to_representer[occurrence['word_id']]
            articles_words_count[occurrence['article_id']][representer] += occurrence['positions_count']
            #if len(positions) == occurrence['positions_count']:
            positions = [int(p) for p in occurrence['positions'].strip().split(',') if p]
            # articles_words_positions[occurrence['article_id']][representer] = positions
            articles_positions[occurrence['article_id']].extend([(p, representer) for p in positions])

        for item_id in articles_positions:
            articles_positions[item_id].sort(key=lambda d: d[0])

        articles_words_count_ngram = defaultdict(lambda: defaultdict(lambda: 0))
        articles_positions_ngram = defaultdict(list)
        for item_id in articles_positions:
            for i in range(0, len(articles_positions[item_id]) - ngram + 1):
                current_positions = articles_positions[item_id][i:i+ngram]
                if current_positions[-1][0] - current_positions[0][0] == ngram - 1:
                    word = tuple(list(map(lambda position: position[1], current_positions)))
                    if word in words:
                        articles_positions_ngram[item_id].append((current_positions[0][0], word))
                        articles_words_count_ngram[item_id][word] += 1
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
                word_string = ', '.join(list(map(lambda x: str(Word.objects.get(id=x)), list(word))))
                logging.debug(' - %-40s %d %.6f (%3d)' % (word_string, count, question_words_weights[word], questions_words_count[word]))
            except Exception as e:
                pass
                #logging.warning('exception during count question words weights')
                #logging.warning(e)
                #logging.warning('question: %s, word: %s' % (question, word))
        return question_words_weights

    def prepare(self, question, is_title):
        logging.info('preparing')
        word_to_representer = TfIdfWeightCalculator.__get_words(question, True)
        words = question.get_ngrams(self.ngram)
        (self.articles_words_count, self.articles_positions) = TfIdfWeightCalculator.__count_articles(words, word_to_representer, is_title, self.ngram)
        (self.articles_words_idf, questions_words_idf) = TfIdfWeightCalculator.__count_idf(self.articles_count, self.articles_words_count)
        self.question_words_weights = TfIdfWeightCalculator.__count_question_words_weights(question, words, self.questions_words_count, questions_words_idf)

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

        filtered_question_words_weights = dict(filter(lambda data: self.articles_words_idf[data[0]] > minimal_word_idf, self.question_words_weights.items()))
        for item_id in self.articles_words_count:
            articles_words_set_weights = []
            words_positions = filter(lambda data: self.articles_words_idf[data[1]] > minimal_word_idf, self.articles_positions[item_id])
            words_positions = self.articles_positions[item_id]
            if sum_neighbors == 0:
                current_words = map(lambda data: data[1], words_positions)
                weights = {}
                for word_id, count in Counter(current_words).items():
                    weights[word_id] = count / len(self.articles_positions[item_id]) * self.articles_words_idf[word_id]
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
                        weights[word_id] = count / sum_neighbors * self.articles_words_idf[word_id]
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
            tools.results_presenter.ResultsPresenter.present(question, articles_id, distances, method, self.debug_top_items, not comparator.ascending_order())
