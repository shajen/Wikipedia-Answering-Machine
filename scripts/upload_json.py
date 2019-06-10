from data.models import *

import argparse
import json
import logging
import os
import shlex
import sys

sys.path.append(os.path.dirname(__file__))

import articles_parser
import logger

def preparse_json(articlesParser, path):
    for line in open(path, 'r'):
        try:
            data = json.loads(line)
            article = Article.objects.create(title=data['title'].strip())
            logging.info('id: %s' % article.id)
            logging.info('title: %s' % article.title)

            logging.debug('forms:')
            for baseText in data['interlinks']:
                logging.debug('%s - %s' % (baseText, data['interlinks'][baseText]))
                articlesParser.addBaseForms(baseText, data['interlinks'][baseText])

        except Exception as e:
            logging.warning('exception durig parse json line')
            logging.warning(line)
            logging.warning(e)

def preparse_polimorfologik(articlesParser, path):
    for line in open(path, 'r'):
        data = line.strip().split('\t')
        articlesParser.addBaseForms(data[1], data[0])

def parse_stop_words(path):
    for line in open(path, 'r'):
        Word.objects.filter(base_form__iexact=line.strip()).update(is_stop_word=True)

def parse_json(articlesParser, path):
    ignoredSections = ['bibliografia', 'linki zewnętrzne', 'zobacz też', 'przypisy', 'uwagi']
    for line in open(path, 'r'):
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
            logging.warning('exception durig parse json line')
            logging.warning(line)
            logging.warning(e)

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("json_articles_file", help="path to json articles file", type=str)
    parser.add_argument("polimorfologik_file", help="path to polimorfologik  file", type=str)
    parser.add_argument("stop_words_file", help="path to stop words file", type=str)
    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args(args)

    logger.configLogger(args.verbose)
    Solution.objects.all().delete()
    Method.objects.all().delete()
    Answer.objects.all().delete()
    Question.objects.all().delete()

    Occurrence.objects.all().delete()
    Word.objects.all().delete()
    Article.objects.all().delete()

    articlesParser = articles_parser.ArticlesParser()
    preparse_polimorfologik(articlesParser, args.polimorfologik_file)
    preparse_json(articlesParser, args.json_articles_file)
    parse_json(articlesParser, args.json_articles_file)
    parse_stop_words(args.stop_words_file)
