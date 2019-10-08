from data.models import *
from django import db
from django.db.models import Sum, Count

import argparse
import logging
import multiprocessing
import os
import queue
import re
import shlex
import subprocess
import sys
sys.path.append(os.path.dirname(__file__))

import calculators.categories_weight_calculator
import calculators.links_weight_calculator
import calculators.tf_idf_weight_calculator
import calculators.weight_comparator
import tools.logger

def update_articles_words_count(is_title):
    logging.info('start update_articles_words_count is_title: %d' % is_title)
    logging.info('reading stop words')
    stop_words = list(map(lambda x: x['id'], Word.objects.filter(is_stop_word = True).values('id')))
    articles = Occurrence.objects.values('article_id').filter(is_title=is_title).exclude(word_id__in=stop_words).annotate(count=Sum('positions_count'))
    logging.info('reading articles words count')
    for article in articles:
        try:
            if is_title:
                Article.objects.filter(id=article['article_id']).update(title_words_count=article['count'])
            else:
                Article.objects.filter(id=article['article_id']).update(content_words_count=article['count'])
        except Exception as e:
            logging.warning('exception during update_articles_words_count:')
            logging.warning(e)
            logging.warning(article)
    logging.info('finished update_articles_words_count')

def resolve_questions(questions_queue, method_name, debug_top_items, neighbors, minimal_word_idf_weights, power_factors):
    tf_idf_calculator = calculators.tf_idf_weight_calculator.TfIdfWeightCalculator(debug_top_items)
    # links_wc = calculators.links_weight_calculator.LinksWeightCalculator(debug_top_items)
    # categories_wc = calculators.categories_weight_calculator.CategoriesWeightCalculator(debug_top_items)

    methods = Method.objects.filter(name__istartswith=method_name)

    def tf_idf_upload_positions(sum_neighbors):
        for minimal_word_idf_weight in minimal_word_idf_weights:
            comparators = []
            m = '%s, mwiw: %.2f' % (method_name, minimal_word_idf_weight)
            for pf in power_factors:
                comparators.append(calculators.weight_comparator.TfIdfWeightComparator(m, sum_neighbors, pf))
            comparators.append(calculators.weight_comparator.CosineWeightComparator(m, sum_neighbors))
            comparators.append(calculators.weight_comparator.EuclideanWeightComparator(m, sum_neighbors))
            comparators.append(calculators.weight_comparator.CityblockWeightComparator(m, sum_neighbors))

            tf_idf_calculator.calculate(q, sum_neighbors, minimal_word_idf_weight, comparators)
        # links_wc.upload_positions(q, method_name, articles_weight)
        # categories_wc.upload_positions(q, method_name, articles_weight)

    while True:
        try:
            q = questions_queue.get(timeout=1)
            logging.info('')
            logging.info('*' * 80)
            logging.info('processing question:')
            logging.info('%d: %s' % (q.id, q.name))

            answers = q.answer_set.all()
            current_solutions_count = Solution.objects.filter(method__in=methods, answer__in=answers).all().count()
            expected_solutions_count = (3 + len(power_factors)) * len(minimal_word_idf_weights) * len(neighbors) * len(answers)

            if current_solutions_count != expected_solutions_count and current_solutions_count > 0:
                logging.info('clearing')
                Solution.objects.filter(method__in=methods, answer__in=answers).delete()
                current_solutions_count = 0

            if current_solutions_count == expected_solutions_count:
                logging.info('skipping')
            else:
                tf_idf_calculator.prepare(q, False)
                for n in neighbors:
                    tf_idf_upload_positions(n)
        except queue.Empty:
            break

def start(questions, num_threads, method_name, debug_top_items, neighbors, minimal_word_idf_weights, power_factors):
    logging.info('start')
    logging.info('threads: %d' % num_threads)
    logging.info('debug_top_items: %d' % debug_top_items)
    logging.info('method_name: %s' % method_name)
    logging.info('neighbors: %s' % neighbors)
    logging.info('minimal_word_idf_weights: %s' % minimal_word_idf_weights)
    logging.info('power_factors: %s' % power_factors)
    db.connections.close_all()

    questions_queue = multiprocessing.Queue()
    for question in questions:
        questions_queue.put(question)

    threads = []
    for i in range(num_threads):
        thread = multiprocessing.Process(target=resolve_questions, args=(questions_queue, method_name, debug_top_items, neighbors, minimal_word_idf_weights, power_factors))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    logging.info('finish')

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", help="threads", type=int, default=1, choices=range(1, 33), metavar="int")
    parser.add_argument('-m', '--method', help="method name to make unique in database", type=str, default='', metavar="method")
    parser.add_argument("-dti", "--debug_top_items", help="print top n items in debug", type=int, default=3, metavar="int")
    parser.add_argument("-mwiw", "--minimal_word_idf_weight", help="use only words with idf weight above", type=float, default=0.0, metavar="float")
    parser.add_argument("-pf", "--power_factor", help="use to sum words weight in count article weight", type=float, default=0.0, metavar="float")
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)
    #update_articles_words_count(True)
    #update_articles_words_count(False)

    questions = list(Question.objects.all())
    dirPath = os.path.dirname(os.path.realpath(__file__))
    commit_hash = subprocess.check_output(['git', '-C', dirPath, 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    commit_datetime = subprocess.check_output(['git', '-C', dirPath, 'log', '-1', '--format=%at']).decode('ascii').strip()
    method_name = 'date: %s, git: %s' % (commit_datetime, commit_hash)
    if args.method:
        method_name = 'name: %s, %s' % (args.method, method_name)
    # neighbors = [0, 1, 3, 5, 10, 15, 20, 30, 50, 100, 150, 200, 250, 500]
    neighbors = [5, 10, 20, 50, 100, 150, 200, 250, 500]
    minimal_word_idf_weights = [args.minimal_word_idf_weight]
    # minimal_word_idf_weights = [0.0, 0.25, 0.5, 0.75, 1.0, 1,5]
    power_factors = [args.power_factor]
    # power_factors = [1.0, 2.0, 3.0, 4.0]
    start(questions, args.threads, method_name, args.debug_top_items, neighbors, minimal_word_idf_weights, power_factors)
