from data.models import *
from django.db.models import Sum, Count

import argparse
import logging
import numpy
import os
import re
import shlex
import subprocess
import sys
import threading
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

def resolve_questions(questions, method_name, debug_top_items):
    tf_idf_wc = calculators.tf_idf_weight_calculator.TfIdfWeightCalculator(debug_top_items)
    cosine_wc = calculators.vector_weight_calculator.CosineVectorWeightCalculator(debug_top_items)
    euclidean_wc = calculators.vector_weight_calculator.EuclideanVectorWeightCalculator(debug_top_items)
    city_wc = calculators.vector_weight_calculator.CityblockVectorWeightCalculator(debug_top_items)
    links_wc = calculators.links_weight_calculator.LinksWeightCalculator(debug_top_items)
    categories_wc = calculators.categories_weight_calculator.CategoriesWeightCalculator(debug_top_items)

    for q in questions:
        logging.info('')
        logging.info('*' * 80)
        logging.info('processing question:')
        logging.info('%d: %s' % (q.id, q.name))

        (question_words_weight, articles_words_weight, articles_weight) = tf_idf_wc.get_weights(q, False, False)
        tf_idf_wc.upload_positions(q, method_name, False, articles_words_weight, articles_weight)
        cosine_wc.upload_positions(q, method_name, False, question_words_weight, articles_words_weight)
        euclidean_wc.upload_positions(q, method_name, False, question_words_weight, articles_words_weight)
        city_wc.upload_positions(q, method_name, False, question_words_weight, articles_words_weight)

        (question_words_neighbors_weights, articles_words_neighbors_weight, articles_weight_neighbors) = tf_idf_wc.get_weights(q, False, True)
        tf_idf_wc.upload_positions(q, method_name, True, articles_words_neighbors_weight, articles_weight_neighbors)
        cosine_wc.upload_positions(q, method_name, True, question_words_neighbors_weights, articles_words_neighbors_weight)
        euclidean_wc.upload_positions(q, method_name, True, question_words_neighbors_weights, articles_words_neighbors_weight)
        city_wc.upload_positions(q, method_name, True, question_words_neighbors_weights, articles_words_neighbors_weight)

        links_wc.upload_positions(q, method_name, articles_weight)
        categories_wc.upload_positions(q, method_name, articles_weight)

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", help="threads", type=int, default=1, choices=range(1, 33), metavar="int")
    parser.add_argument('-m', '--method', help="method name to make unique in database", type=str, default='', metavar="method")
    parser.add_argument("-dti", "--debug_top_items", help="print top n items in debug", type=int, default=3, metavar="int")
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)
    logging.info('start')
    logging.info('threads: %d' % args.threads)
    logging.info('debug_top_items: %d' % args.debug_top_items)
    #update_articles_words_count(True)
    #update_articles_words_count(False)

    threads = []
    questions = list(Question.objects.all())
    questions_sets = numpy.array_split(questions, args.threads)
    dirPath = os.path.dirname(os.path.realpath(__file__))
    commit_hash = subprocess.check_output(['git', '-C', dirPath, 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    commit_datetime = subprocess.check_output(['git', '-C', dirPath, 'log', '-1', '--format=%at']).decode('ascii').strip()
    method_name = 'date: %s, git: %s' % (commit_datetime, commit_hash)
    if args.method:
        method_name = 'name: %s, %s' % (args.method, method_name)
    logging.info('method_name: %s' % method_name)
    for questions_set in questions_sets:
        thread = threading.Thread(target=resolve_questions, args=(questions_set, method_name, args.debug_top_items))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    logging.info('finish')
