from collections import defaultdict, deque
from data.models import *
import calculators.weight_calculator
import logging
import math
import re

class TfIdfWeightCalculator(calculators.weight_calculator.WeightCalculator):
    def __init__(self, debug_top_items):
        super().__init__(debug_top_items)
        logging.info('start parsing questions')
        self.questions_words_count = defaultdict(lambda: 0)
        for question in Question.objects.all():
            for word in set(TfIdfWeightCalculator.__parse_question(question).values()):
                self.questions_words_count[word] += 1
        logging.info('finish parsing questions')

    def __parse_question(question, debug=False):
        words = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', question.name)
        words = Word.objects.filter(changed_form__in=words).order_by('changed_form').values('id', 'is_stop_word', 'base_form', 'changed_form')
        base_form_to_id = {}
        for word in words:
            if list(filter(lambda w: w['changed_form'] == word['changed_form'] and w['is_stop_word'], words)):
                continue
            first_base_form = list(filter(lambda w: w['changed_form'] == word['changed_form'] and not w['is_stop_word'], words))[0]
            if debug:
                logging.debug('%s - %s' % (first_base_form['base_form'], word['base_form']))
            base_form_to_id[word['base_form']] = first_base_form['id']
        return base_form_to_id

    def __count_articles(question_words_changed_form_to_base_form, is_title):
        articles_words_count = defaultdict(lambda: defaultdict(lambda: 0))
        # articles_words_positions = defaultdict(defaultdict)
        articles_positions = defaultdict(list)
        occurrences = Occurrence.objects.filter(word_id__in=question_words_changed_form_to_base_form.keys(), is_title=is_title).values('article_id', 'word_id', 'positions_count', 'positions')
        for occurrence in occurrences:
            base_form_id = question_words_changed_form_to_base_form[occurrence['word_id']]
            articles_words_count[occurrence['article_id']][base_form_id] += occurrence['positions_count']
            #if len(positions) == occurrence['positions_count']:
            positions = [int(p) for p in occurrence['positions'].strip().split(',') if p]
            # articles_words_positions[occurrence['article_id']][base_form_id] = positions
            articles_positions[occurrence['article_id']].extend([(p, base_form_id) for p in positions])

        for item_id in articles_positions:
            articles_positions[item_id].sort(key=lambda d: d[0])
        return (articles_words_count, articles_positions)

    def __count_idf(articles_title_count, articles_words_count):
        words_articles_count = defaultdict(lambda: 0)
        for item_id in articles_words_count:
            for word_id in articles_words_count[item_id]:
                words_articles_count[word_id] += 1

        articles_words_idf = {}
        questions_words_idf = {}
        for word_id in words_articles_count:
            articles_words_idf[word_id] = math.log(len(articles_title_count) / words_articles_count[word_id])
            questions_words_idf[word_id] = articles_words_idf[word_id]

        # for word in set(question_words_changed_form_to_base_form.values()):
        #     questions_words_idf[word] = math.log(Question.objects.count() / questions_words_count[word])
        return (articles_words_idf, questions_words_idf)

    def __count_question_words_weights(question, question_words_changed_form_to_base_form, questions_words_count, questions_words_idf):
        logging.debug('question words weights')
        question_words_weights = {}
        for word in set(question_words_changed_form_to_base_form.values()):
            try:
                tf = 1.0 / len(set(question_words_changed_form_to_base_form.values()))
                question_words_weights[word] = tf * questions_words_idf[word]
                logging.debug(' - %-40s %.6f (%3d)' % (Word.objects.get(id=word), question_words_weights[word], questions_words_count[word]))
            except Exception as e:
                logging.warning('exception during count question words weights')
                logging.warning(e)
                logging.warning('question: %s, word: %s' % (question, word))
        return question_words_weights

    def prepare(self, question, is_title):
        question_words_base_form_to_id = TfIdfWeightCalculator.__parse_question(question, True)
        question_words_changed_form_to_base_form = {}
        for word in Word.objects.filter(base_form__in=question_words_base_form_to_id.keys(), is_stop_word=False).values('id', 'base_form'):
            question_words_changed_form_to_base_form[word['id']] = question_words_base_form_to_id[word['base_form']]

        (self.articles_words_count, self.articles_positions) = TfIdfWeightCalculator.__count_articles(question_words_changed_form_to_base_form, is_title)
        (self.articles_words_idf, questions_words_idf) = TfIdfWeightCalculator.__count_idf(self.articles_title_count, self.articles_words_count)
        self.question_words_weights = TfIdfWeightCalculator.__count_question_words_weights(question, question_words_changed_form_to_base_form, self.questions_words_count, questions_words_idf)

    def __count_tf_idf(self, question, sum_neighbors, minimal_word_idf, comparator):
        articles_words_weights = defaultdict(defaultdict)
        articles_weight = defaultdict()

        for item_id in self.articles_words_count:
            max_weight = 0.0
            counter = defaultdict(lambda: 0)
            words_positions = list(filter(lambda data: self.articles_words_idf[data[1]] > minimal_word_idf, self.articles_positions[item_id]))
            current_words = deque()
            articles_words_set_weights = []

            item_words_count = (len(self.articles_positions[item_id]) if sum_neighbors == 0 else (sum_neighbors+1))
            if not words_positions:
                continue
            for data in words_positions:
                current_words.append(data)
                counter[data[1]] += 1
                while current_words[0][0] + (10**9 if sum_neighbors == 0 else sum_neighbors) < current_words[-1][0]:
                    pop_data = current_words.popleft()
                    counter[pop_data[1]] -= 1
                    if counter[pop_data[1]] == 0:
                        del counter[pop_data[1]]

                weights = {}
                for word_id, count in counter.items():
                    weights[word_id] = count / item_words_count * self.articles_words_idf[word_id]
                articles_words_set_weights.append(weights)

            try:
                filtered_question_words_weights = dict(filter(lambda data: self.articles_words_idf[data[0]] > minimal_word_idf, self.question_words_weights.items()))
                (best_weight, best_words_weights) = comparator.get_best_score(filtered_question_words_weights, articles_words_set_weights)
                for word_id in best_words_weights:
                    articles_words_weights[item_id][word_id] = best_words_weights[word_id]
                articles_weight[item_id] = best_weight
            except ValueError as e:
                logging.warn('exception during count articles weight')
                logging.warn(e)
                logging.warning('question: %s, article id: %s, fqww: %d, awsw: %d' % (question, item_id, len(filtered_question_words_weights), len(articles_words_set_weights)))

        return (articles_words_weights, articles_weight)

    def calculate(self, question, sum_neighbors, minimal_word_idf_weight, comparators):
        for comparator in comparators:
            logging.info('')
            logging.info('method %s' % comparator.method())
            (articles_words_weight, articles_weight) = self.__count_tf_idf(question, sum_neighbors, minimal_word_idf_weight, comparator)
            positions = self._count_positions(question, articles_words_weight, articles_weight, comparator.ascending_order(), Article.objects, Word.objects)
            self._upload_positions(positions, comparator.method())
