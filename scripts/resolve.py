from data.models import *
from django import db

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
import calculators.word2vec_weight_calculator
import tools.logger

def resolve_questions_tf_idf(questions_queue, method, is_title, ngram, debug_top_items, tfidf_models, vector_models, neighbors, minimal_word_idf_weights, power_factors):
    tf_idf_calculator = calculators.tf_idf_weight_calculator.TfIdfWeightCalculator(debug_top_items, ngram)
    # links_wc = calculators.links_weight_calculator.LinksWeightCalculator(debug_top_items)
    # categories_wc = calculators.categories_weight_calculator.CategoriesWeightCalculator(debug_top_items)

    while True:
        try:
            question = questions_queue.get(timeout=1)
            logging.info('')
            logging.info('*' * 80)
            logging.info('processing question:')
            logging.info('%d: %s' % (question.id, question.name))

            prepared = False
            for neighbor in neighbors:
                for minimal_word_idf_weight in minimal_word_idf_weights:
                    comparators = []
                    method_name = '%s, mwiw: %.2f' % (method, minimal_word_idf_weight)
                    if tfidf_models:
                        for power_factor in power_factors:
                            comparators.append(calculators.weight_comparator.TfIdfWeightComparator(method_name, neighbor, power_factor))
                    if vector_models:
                        comparators.append(calculators.weight_comparator.CosineWeightComparator(method_name, neighbor))
                        comparators.append(calculators.weight_comparator.EuclideanWeightComparator(method_name, neighbor))
                        comparators.append(calculators.weight_comparator.CityblockWeightComparator(method_name, neighbor))

                    comparators = list(filter(lambda comparator: not comparator.has_already_solutions(question), comparators))
                    if comparators:
                        if not prepared:
                            tf_idf_calculator.prepare(question, is_title)
                            prepared = True
                        tf_idf_calculator.calculate(question, neighbor, minimal_word_idf_weight, comparators)
                # links_wc.upload_positions(q, method, articles_weight)
                # categories_wc.upload_positions(q, method, articles_weight)
        except queue.Empty:
            break

def start_tfidf(questions, num_threads, method_name, is_title, ngram, debug_top_items, tfidf_models, vector_models, neighbors, minimal_word_idf_weights, power_factors):
    db.connections.close_all()
    logging.info('neighbors: %s' % neighbors)
    logging.info('minimal_word_idf_weights: %s' % minimal_word_idf_weights)
    logging.info('power_factors: %s' % power_factors)
    method_name = '%s, ngram: %d' % (method_name, ngram)
    logging.info('method_name: %s' % method_name)
    questions_queue = multiprocessing.Queue()
    for question in questions:
        questions_queue.put(question)

    try:
        threads = []
        for i in range(num_threads):
            thread = multiprocessing.Process(target=resolve_questions_tf_idf, args=(questions_queue, method_name, is_title, ngram, debug_top_items, tfidf_models, vector_models, neighbors, minimal_word_idf_weights, power_factors))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logging.info('stopping threads')
        for thread in threads:
            thread.terminate()

def resolve_questions_word2vec(questions_queue, method_name, is_title, debug_top_items, topn, word2vec_model):
    word2vec_calculator = calculators.word2vec_weight_calculator.Word2VecWeightCalculator(debug_top_items, word2vec_model)
    while True:
        try:
            question = questions_queue.get(timeout=1)
            if not word2vec_calculator.has_already_solutions(question, method_name):
                word2vec_calculator.calculate(question, method_name, is_title, topn)
        except queue.Empty:
            break

def start_word2vec(questions, num_threads, method_name, is_title, debug_top_items, topn, word2vec_model):
    db.connections.close_all()
    logging.getLogger("gensim").setLevel(logging.WARNING)
    logging.getLogger("smart_open.smart_open_lib").setLevel(logging.WARNING)
    logging.info('topn: %s' % topn)
    method_name = '%s, topn: %d, type: word2vec' % (method_name, topn)
    logging.info('method_name: %s' % method_name)
    questions_queue = multiprocessing.Queue()
    for question in questions:
        questions_queue.put(question)

    threads = []
    try:
        for i in range(num_threads):
            thread = multiprocessing.Process(target=resolve_questions_word2vec, args=(questions_queue, method_name, is_title, debug_top_items, topn, word2vec_model))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logging.info('stopping threads')
        for thread in threads:
            thread.terminate()

def start(questions, num_threads, method_name, is_title, ngram, debug_top_items, topn, tfidf_models, vector_models, word2vec_model, neighbors, minimal_word_idf_weights, power_factors):
    logging.info('start')
    logging.info('threads: %d' % num_threads)
    logging.info('debug_top_items: %d' % debug_top_items)
    if tfidf_models or vector_models:
        start_tfidf(questions, num_threads, method_name, is_title, ngram, debug_top_items, tfidf_models, vector_models, neighbors, minimal_word_idf_weights, power_factors)
    if word2vec_model:
        start_word2vec(questions, num_threads, method_name, is_title, debug_top_items, topn, word2vec_model)
    logging.info('finish')

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", help="threads", type=int, default=1, choices=range(1, 129), metavar="int")
    parser.add_argument("-q", "--questions", help="solve first n questions", type=int, default=10**9, metavar="n")
    parser.add_argument("-T", "--title", help="calculate based on articles title not content", action='store_true')
    parser.add_argument('-n', '--ngram', help="use ngram mode", type=int, default=1, metavar="ngram")
    parser.add_argument("-tm", "--tfidf_models", help="use td-idf models", action='store_true')
    parser.add_argument("-vm", "--vector_models", help="use vector models", action='store_true')
    parser.add_argument("-w2vm", "--word2vec_model", help="use word2vec model", type=str, default='', metavar="file")
    parser.add_argument('-m', '--method', help="method name to make unique in database", type=str, default='', metavar="method")
    parser.add_argument("-dti", "--debug_top_items", help="print top n items in debug", type=int, default=3, metavar="int")
    parser.add_argument("-tn", "--topn", help="use n nearest words in word2vec model", type=int, default=10, metavar="n")
    parser.add_argument("-N", "--neighbors", help="count tf-idf in every n neighboring words tuple in articles", type=str, default='0', metavar="0,10,20")
    parser.add_argument("-mwiw", "--minimal_word_idf_weights", help="use only words with idf weight above", type=str, default='0.0', metavar="0.0,1.6,3.2")
    parser.add_argument("-pf", "--power_factors", help="use to sum words weight in count article weight", type=str, default='4', metavar="1,2,3,4")
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)

    questions = list(Question.objects.all())[:args.questions]
    dirPath = os.path.dirname(os.path.realpath(__file__))
    commit_hash = subprocess.check_output(['git', '-C', dirPath, 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    commit_datetime = subprocess.check_output(['git', '-C', dirPath, 'log', '-1', '--format=%at']).decode('ascii').strip()
    method_name = 'git: %s_%s, title: %d' % (commit_datetime, commit_hash, args.title)
    if args.method:
        method_name = 'name: %s, %s' % (args.method, method_name)
    neighbors = list(map(lambda x: int(x), args.neighbors.split(',')))
    minimal_word_idf_weights = list(map(lambda x: float(x), args.minimal_word_idf_weights.split(',')))
    power_factors = list(map(lambda x: int(x), args.power_factors.split(',')))
    start(questions, args.threads, method_name, args.title, args.ngram, args.debug_top_items, args.topn, args.tfidf_models, args.vector_models, args.word2vec_model, neighbors, minimal_word_idf_weights, power_factors)
