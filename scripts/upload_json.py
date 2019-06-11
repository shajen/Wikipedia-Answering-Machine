from data.models import *
import multiprocess

import argparse
import json
import logging
import os
import shlex
import sys
from functools import reduce

sys.path.append(os.path.dirname(__file__))

import articles_parser
import logger

def preparse_json(batch_size, line):
    try:
        articlesParser = articles_parser.ArticlesParser(batch_size)
        words = []
        data = json.loads(line)
        logging.debug('%s' % (data['title']))

        for baseText in data['interlinks']:
            logging.debug('%s - %s' % (baseText, data['interlinks'][baseText]))
            words.extend(articlesParser.addBaseForms(baseText, data['interlinks'][baseText]))

        Word.objects.bulk_create(words, ignore_conflicts=True, batch_size=batch_size)
        return Article(title=data['title'].strip())
    except Exception as e:
        logging.warning('exception durig preparse json line:')
        logging.warning(e)
        logging.warning(line)

def preparse_polimorfologik(line):
    articlesParser = articles_parser.ArticlesParser(0)
    data = line.strip().split('\t')
    return Word(base_form=data[1], changed_form=data[0])

def parse_stop_words(line):
    Word.objects.filter(base_form__iexact=line.strip()).update(is_stop_word=True)

def parse_json(batch_size, line):
    articlesParser = articles_parser.ArticlesParser(batch_size)
    ignoredSections = ['bibliografia', 'linki zewnętrzne', 'zobacz też', 'przypisy', 'uwagi']
    try:
        data = json.loads(line)
        title = data['title'].strip()
        text = ''
        links = []

        logging.info('title: %s' % title)
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
        logging.error('exception durig parse json line:')
        logging.error(e)
        logging.error(line)

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
    pool = multiprocess.Pool(args.threads)

    Solution.objects.all().delete()
    Method.objects.all().delete()
    Answer.objects.all().delete()
    Question.objects.all().delete()

    Occurrence.objects.all().delete()
    Word.objects.all().delete()
    Article.objects.all().delete()

    words = pool.map(preparse_polimorfologik, list(open(args.polimorfologik_file, 'r')))
    Word.objects.bulk_create(words, ignore_conflicts=True, batch_size=args.batch_size)

    articles = pool.map(partial(preparse_json, args.batch_size), list(open(args.json_articles_file, 'r')))
    articles = [x for x in articles if x is not None]
    Article.objects.bulk_create(articles, ignore_conflicts=True, batch_size=args.batch_size)

    pool.map(partial(parse_json, args.batch_size), list(open(args.json_articles_file, 'r')))
    pool.map(parse_stop_words, list(open(args.stop_words_file, 'r')))
