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

def resolve_questions(questions, debug_top_articles, is_title):
    wc = tools.weight_calculator.WeightCalculator(debug_top_articles)
    for q in questions:
        wc.count_tf_idf(q, is_title)

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", help="threads", type=int, default=1, choices=range(1, 33), metavar="int")
    parser.add_argument("-dta", "--debug_top_articles", help="print top n articles in debug", type=int, default=3, metavar="int")
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)
    logging.info('start')
    logging.info('debug_top_articles: %d' % args.debug_top_articles)
    logging.info('threads: %d' % args.threads)
    #update_articles_words_count(True)
    #update_articles_words_count(False)

    threads = []
    questions = list(Question.objects.all()[:16])
    questions_sets = numpy.array_split(questions, args.threads)
    for questions_set in questions_sets:
        thread = threading.Thread(target=resolve_questions, args=(questions_set, args.debug_top_articles, False))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    logging.info('finish')
