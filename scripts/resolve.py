from data.models import *
from django.db.models import Sum, Count

import logging
import os
import sys
import shlex
import argparse

sys.path.append(os.path.dirname(__file__))

import tools.logger

def update_articles_count():
    logging.info('reading stop words')
    stop_words = list(map(lambda x: x['id'], Word.objects.filter(is_stop_word = True).values('id')))
    articles = Occurrence.objects.values('article_id').filter(is_title=False).exclude(word_id__in=stop_words).annotate(count=Sum('positions_count'))
    logging.info('reading articles length')
    articles_count = {}
    i = 0
    for article in articles:
        if i % 100000 == 0:
            logging.info('reading articles %.2f%%' % (100.0 * i / len(articles)))
            logging.info(article)
        i += 1
        if article['count']:
            articles_count[article['article_id']] = article['count']
    logging.info('finished articles count')

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)
    logging.info('start')
    update_articles_count()
    logging.info('finish')
