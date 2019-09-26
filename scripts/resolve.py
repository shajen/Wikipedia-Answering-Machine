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
import calculators.vector_weight_calculator
import calculators.weight_calculator
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

def resolve_questions(questions_queue, method_name, debug_top_items, minimal_word_idf_weight, power_factor):
    tf_idf_wc = calculators.tf_idf_weight_calculator.TfIdfWeightCalculator(debug_top_items)
    # cosine_wc = calculators.vector_weight_calculator.CosineVectorWeightCalculator(debug_top_items)
    # euclidean_wc = calculators.vector_weight_calculator.EuclideanVectorWeightCalculator(debug_top_items)
    # city_wc = calculators.vector_weight_calculator.CityblockVectorWeightCalculator(debug_top_items)
    # links_wc = calculators.links_weight_calculator.LinksWeightCalculator(debug_top_items)
    # categories_wc = calculators.categories_weight_calculator.CategoriesWeightCalculator(debug_top_items)

    methods = Method.objects.filter(name__istartswith=method_name)

    def tf_idf_upload_positions(sum_neighbors):
        (question_words_weight, articles_words_weight, articles_weight) = tf_idf_wc.get_weights(q, False, sum_neighbors, minimal_word_idf_weight, power_factor)
        tf_idf_wc.upload_positions(q, method_name, sum_neighbors, articles_words_weight, articles_weight)
        # cosine_wc.upload_positions(q, method_name, sum_neighbors, question_words_weight, articles_words_weight)
        # euclidean_wc.upload_positions(q, method_name, sum_neighbors, question_words_weight, articles_words_weight)
        # city_wc.upload_positions(q, method_name, sum_neighbors, question_words_weight, articles_words_weight)
        # links_wc.upload_positions(q, method_name, articles_weight)
        # categories_wc.upload_positions(q, method_name, articles_weight)

    while True:
        try:
            q = questions_queue.get(timeout=1)
            logging.info('')
            logging.info('*' * 80)
            logging.info('processing question:')
            logging.info('%d: %s' % (q.id, q.name))

            neighbors = [0, 1, 3, 5, 10, 15, 20, 30, 50, 100, 150, 200, 250, 500]
            # neighbors = [5, 10, 20, 50, 100, 150, 200, 250, 500]

            answers = q.answer_set.all()
            current_solutions_count = Solution.objects.filter(method__in=methods, answer__in=answers).all().count()
            expected_solutions_count = 4 * len(neighbors) * len(answers)

            if current_solutions_count != expected_solutions_count and current_solutions_count > 0:
                logging.info('clearing')
                Solution.objects.filter(method__in=methods, answer__in=answers).delete()
                current_solutions_count = 0

            if current_solutions_count == expected_solutions_count:
                logging.info('skipping')
            else:
                tf_idf_wc.prepare(q, False)
                for n in neighbors:
                    tf_idf_upload_positions(n)
        except queue.Empty:
            break

def start(questions, num_threads, method_name, debug_top_items, minimal_word_idf_weight, power_factor):
    logging.info('start')
    logging.info('threads: %d' % num_threads)
    logging.info('debug_top_items: %d' % debug_top_items)
    logging.info('minimal_word_idf_weight: %.2f' % minimal_word_idf_weight)
    logging.info('power_factor: %.2f' % power_factor)
    logging.info('method_name: %s' % method_name)
    db.connections.close_all()

    questions_queue = multiprocessing.Queue()
    for question in questions:
        questions_queue.put(question)

    threads = []
    for i in range(num_threads):
        thread = multiprocessing.Process(target=resolve_questions, args=(questions_queue, method_name, debug_top_items, minimal_word_idf_weight, power_factor))
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
    method_name = 'date: %s, git: %s, mwiw: %.2f, pf: %.2f' % (commit_datetime, commit_hash, args.minimal_word_idf_weight, args.power_factor)
    if args.method:
        method_name = 'name: %s, %s' % (args.method, method_name)
    start(questions, args.threads, method_name, args.debug_top_items, args.minimal_word_idf_weight, args.power_factor)
