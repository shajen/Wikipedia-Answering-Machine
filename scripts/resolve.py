from data.models import *
from django.db.models import Sum, Count

import logging
import os
import sys
import shlex
import argparse
import re
import numpy
import threading
import subprocess

sys.path.append(os.path.dirname(__file__))

import tools.logger
import tools.weight_calculator

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

def resolve_questions(questions, method_name, debug_top_items, is_title):
    method_tf_idf, created = Method.objects.get_or_create(name=method_name + ", type: tf_idf")
    method_tf_idf_neighbours, created = Method.objects.get_or_create(name=method_name + ", type: tf_idf_neighbours")
    method_link, created = Method.objects.get_or_create(name=method_name + ", type: link")
    method_reverse_link, created = Method.objects.get_or_create(name=method_name + ", type: reverse_link")
    method_category, created = Method.objects.get_or_create(name=method_name + ", type: category")
    wc = tools.weight_calculator.WeightCalculator(debug_top_items)
    for q in questions:
        logging.info('')
        logging.info('*' * 80)
        logging.info('processing question:')
        logging.info('%d: %s' % (q.id, q.name))

        logging.info('')
        logging.info('tf-idf')
        articles_words_weight = wc.count_tf_idf(q, is_title, False)
        articles_words_positions = wc.count_positions(q, articles_words_weight, 3, Article.objects, Word.objects)
        logging.info(articles_words_positions)

        logging.info('')
        logging.info('tf-idf neighbors')
        articles_words_weight_neighbors = wc.count_tf_idf(q, is_title, True)
        articles_words_positions_neighbors = wc.count_positions(q, articles_words_weight_neighbors, 1.7, Article.objects, Word.objects)
        logging.info(articles_words_positions_neighbors)

        articles_weight = wc.count_weights(articles_words_weight, 3)
        logging.info('')
        logging.info('links')
        (articles_links_articles_weight, articles_reverse_links_articles_weight) = wc.count_articles_links_weight(articles_weight, 10)
        articles_links_articles_positions = wc.count_positions(q, articles_links_articles_weight, 0, Article.objects, Article.objects)
        logging.info(articles_links_articles_positions)
        logging.info('')
        logging.info('reverse links')
        articles_reverse_links_articles_positions = wc.count_positions(q, articles_reverse_links_articles_weight, 0, Article.objects, Article.objects)
        logging.info(articles_reverse_links_articles_positions)

        logging.info('')
        logging.info('categories')
        categories_articles_weight = wc.count_categories_weight(articles_weight, 10)
        articles_categories_weight = wc.count_articles_categories_weight(categories_articles_weight, 10, 0)
        articles_categories_positions = wc.count_positions(q, articles_categories_weight, 0, Article.objects, Category.objects)
        logging.info(articles_categories_positions)

        for (answer, position) in articles_words_positions:
           Solution.objects.create(answer=answer, position=position, method=method_tf_idf)
        for (answer, position) in articles_words_positions_neighbors:
           Solution.objects.create(answer=answer, position=position, method=method_tf_idf_neighbours)
        for (answer, position) in articles_links_articles_positions:
            Solution.objects.create(answer=answer, position=position, method=method_link)
        for (answer, position) in articles_reverse_links_articles_positions:
            Solution.objects.create(answer=answer, position=position, method=method_reverse_link)
        for (answer, position) in articles_categories_positions:
            Solution.objects.create(answer=answer, position=position, method=method_category)

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", help="threads", type=int, default=1, choices=range(1, 33), metavar="int")
    parser.add_argument("--title", help="count weight in title", action='store_true')
    parser.add_argument('-m', '--method', help="method name to make unique in database", type=str, default='', metavar="method")
    parser.add_argument("-dti", "--debug_top_items", help="print top n items in debug", type=int, default=3, metavar="int")
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)
    logging.info('start')
    logging.info('threads: %d' % args.threads)
    logging.info('title: %d' % args.title)
    logging.info('debug_top_items: %d' % args.debug_top_items)
    #update_articles_words_count(True)
    #update_articles_words_count(False)

    threads = []
    questions = list(Question.objects.all())
    questions_sets = numpy.array_split(questions, args.threads)
    dirPath = os.path.dirname(os.path.realpath(__file__))
    commit_hash = subprocess.check_output(['git', '-C', dirPath, 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    commit_datetime = subprocess.check_output(['git', '-C', dirPath, 'log', '-1', '--format=%at']).decode('ascii').strip()
    method_name = 'title: %d, date: %s, git: %s' % (args.title, commit_datetime, commit_hash)
    if args.method:
        method_name = 'name: %s, %s' % (args.method, method_name)
    logging.info('method_name: %s' % method_name)
    for questions_set in questions_sets:
        thread = threading.Thread(target=resolve_questions, args=(questions_set, method_name, args.debug_top_items, args.title))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    logging.info('finish')
