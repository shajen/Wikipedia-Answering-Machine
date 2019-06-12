from data.models import *
import multiprocess

import argparse
import json
import logging
import os
import shlex
import sys
from functools import reduce, partial

sys.path.append(os.path.dirname(__file__))

import articles_parser
import logger

def preparse_article_callback(batch_size, line):
    try:
        articlesParser = articles_parser.ArticlesParser(batch_size)
        words = []
        data = json.loads(line)
        logging.debug('%s' % (data['title']))

        for baseText in data['interlinks']:
            logging.debug('%s - %s' % (baseText, data['interlinks'][baseText]))
            words.extend(articlesParser.addBaseForms(baseText, data['interlinks'][baseText]))

        return (Article(title=data['title'].strip()), words)
    except Exception as e:
        logging.warning('exception durig preparse_article_callback:')
        logging.warning(e)
        logging.warning(line)
        return (None, [])

def preparse_articles(batch_size, file, pool):
    logging.info('preparse articles start')
    data = pool.map(partial(preparse_article_callback, batch_size), list(open(file, 'r')))
    logging.info('inserting %d words' % reduce(lambda x, y: x + y, [len(w) for (a, w) in data]))
    words = []
    for (a, w) in data:
        words.extend(w)
        if len(words) >= batch_size:
            Word.objects.bulk_create(words, ignore_conflicts=True)
            words = []
    Word.objects.bulk_create(words, ignore_conflicts=True)
    articles = [a for (a, w) in data if a is not None]
    logging.info('inserting %d articles' % len(articles))
    Article.objects.bulk_create(articles, ignore_conflicts=True, batch_size=batch_size)
    logging.info('finish')

def preparse_polimorfologik(batch_size, file):
    logging.info('preparse polimorfologik start')
    words = []
    articlesParser = articles_parser.ArticlesParser(batch_size)
    for line in list(open(file, 'r')):
        data = line.strip().split('\t')
        words.append(Word(base_form=data[1], changed_form=data[0]))
        if len(words) >= batch_size:
            logging.debug('inserting %d rows' % len(words))
            try:
                Word.objects.bulk_create(words, ignore_conflicts=True)
                logging.debug('finish')
            except Exception as e:
                logging.warning('exeption during insert words:')
                logging.warning(e)
                logging.warning(words)
            words = []
    logging.info('finish')

def parse_stop_words(file):
    logging.info('parse stop words start')
    stop_words = [line.strip().lower() for line in open(file, 'r')]
    Word.objects.filter(base_form__in=stop_words).update(is_stop_word=True)
    logging.info('finish')

def parse_articles_callback(batch_size, line):
    articlesParser = articles_parser.ArticlesParser(batch_size)
    ignoredSections = ['bibliografia', 'linki zewnętrzne', 'zobacz też', 'przypisy', 'uwagi']
    try:
        data = json.loads(line)
        title = data['title'].strip()
        text = ''
        links = []

        logging.debug('title: %s' % title)
        logging.debug('sections:')
        for i in range(len(data['section_titles'])):
            sectionName = data['section_titles'][i]
            sectionText = data['section_texts'][i]
            if sectionName.lower().strip() in ignoredSections:
                continue
            logging.debug(sectionName)
            logging.debug(sectionText)
            text += sectionText

        for baseText in data['interlinks']:
            links.append(baseText)

        articlesParser.parseArticle(title, text, links)
        # print(json.dumps(data, indent=4, sort_keys=True))
    except Exception as e:
        logging.warning('exception durig parse_articles_callback:')
        logging.warning(e)
        logging.warning(line)

def parse_articles(batch_size, file, pool):
    logging.info('parse articles start')
    pool.map(partial(parse_articles_callback, batch_size), list(open(file, 'r')))
    logging.info('parse stop words start')

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("json_articles_file", help="path to json articles file", type=str)
    parser.add_argument("polimorfologik_file", help="path to polimorfologik  file", type=str)
    parser.add_argument("stop_words_file", help="path to stop words file", type=str)
    parser.add_argument("-t", "--threads", help="threads", type=int, default=1, choices=range(1, 33), metavar="int")
    parser.add_argument("-b", "--batch_size", help="batch_size", type=int, default=10000, metavar="int")
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args(args)

    logger.configLogger(args.verbose)
    logging.info('start')
    logging.info('threads: %d' % args.threads)
    logging.info('batch size: %d' % args.batch_size)

    Solution.objects.all().delete()
    Method.objects.all().delete()
    Answer.objects.all().delete()
    Question.objects.all().delete()
    Occurrence.objects.all().delete()
    Word.objects.all().delete()
    Article.objects.all().delete()

    pool = multiprocess.Pool(args.threads)
    preparse_polimorfologik(args.batch_size, args.polimorfologik_file)
    preparse_articles(args.batch_size, args.json_articles_file, pool)
    parse_articles(args.batch_size, args.json_articles_file, pool)
    parse_stop_words(args.stop_words_file)
