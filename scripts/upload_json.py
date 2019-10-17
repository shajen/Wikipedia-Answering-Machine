from data.models import *
import multiprocess

import argparse
import json
import logging
import os
import shlex
import sys
import smart_open
from functools import reduce, partial

sys.path.append(os.path.dirname(__file__))

import tools.articles_parser
import tools.logger

def preparse_article_callback(batch_size, category_tag, line):
    try:
        articlesParser = tools.articles_parser.ArticlesParser(batch_size, category_tag)
        words = []
        data = json.loads(line)
        title = data['title'].strip().lower()
        logging.debug('%s' % (title))

        for baseText in data['interlinks']:
            logging.debug('%s - %s' % (baseText, data['interlinks'][baseText]))
            words.extend(articlesParser.addBaseForms(baseText, data['interlinks'][baseText]))

        if title.startswith(category_tag):
            title = title[len(category_tag):]
            return (None, Category(title=title), words)
        else:
            return (Article(title=title), None, words)
    except Exception as e:
        logging.warning('exception durig preparse_article_callback:')
        logging.warning(e)
        logging.warning(line)
        return (None, None, [])

def preparse_articles(batch_size, file, category_tag, first_n_lines, pool):
    logging.info('preparse articles start')
    lines = list(smart_open.open(file, 'r'))
    if first_n_lines > 0:
        lines = lines[:first_n_lines]
    data = pool.map(partial(preparse_article_callback, batch_size, category_tag), lines)
    logging.info('inserting %d words' % reduce(lambda x, y: x + y, [len(w) for (a, c, w) in data]))
    words = []
    for (a, c, w) in data:
        words.extend(w)
        if len(words) >= batch_size:
            Word.objects.bulk_create(words, ignore_conflicts=True)
            words = []
    Word.objects.bulk_create(words, ignore_conflicts=True)
    articles = [a for (a, c, w) in data if a is not None]
    logging.info('inserting %d articles' % len(articles))
    Article.objects.bulk_create(articles, ignore_conflicts=True, batch_size=batch_size)
    categories = [c for (a, c, w) in data if c is not None]
    logging.info('inserting %d categories' % len(categories))
    Category.objects.bulk_create(categories, ignore_conflicts=True, batch_size=batch_size)

    logging.info('finish')

def insert_words(words):
    logging.debug('inserting %d words' % len(words))
    try:
        Word.objects.bulk_create(words, ignore_conflicts=True)
        logging.debug('finish')
    except Exception as e:
        logging.warning('exeption during insert words:')
        logging.warning(e)
        logging.warning(words)

def preparse_polimorfologik(batch_size, file, category_tag, first_n_lines):
    logging.info('preparse polimorfologik start')
    words = []
    articlesParser = tools.articles_parser.ArticlesParser(batch_size, category_tag)
    lines = list(open(file, 'r'))
    if first_n_lines > 0:
        lines = lines[:first_n_lines]
    lines = list(set([tuple(line.strip().lower().split('\t')[:2]) for line in lines]))
    logging.info('inserting %d words\n' % len(lines))
    for (changed_form, base_form) in lines:
        words.append(Word(base_form=base_form, changed_form=changed_form))
        if len(words) >= batch_size:
            insert_words(words)
            words = []
    insert_words(words)
    logging.info('finish')

def parse_stop_words(file, first_n_lines):
    logging.info('parse stop words start')
    lines = list(open(file, 'r'))
    if first_n_lines > 0:
        lines = lines[:first_n_lines]
    stop_words = [line.strip().lower() for line in lines]
    Word.objects.filter(base_form__in=stop_words).update(is_stop_word=True)
    logging.info('finish')

def parse_articles_callback(batch_size, category_tag, redirect_tag, line):
    articlesParser = tools.articles_parser.ArticlesParser(batch_size, category_tag)
    ignoredSections = ['bibliografia', 'linki zewnętrzne', 'zobacz też', 'przypisy', 'uwagi']
    try:
        data = json.loads(line)
        title = data['title'].strip().lower()
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
            links.append(baseText.strip().lower())

        if title.startswith(category_tag):
            title = title[len(category_tag):]
            articlesParser.parseCategory(title, text, links)
        elif text.lower().startswith(redirect_tag):
            articlesParser.parseRedirect(title, text, links)
        else:
            articlesParser.parseArticle(title, text, links)
        # print(json.dumps(data, indent=4, sort_keys=True))
    except Exception as e:
        logging.warning('exception durig parse_articles_callback:')
        logging.warning(e)
        logging.warning(line)

def parse_articles(batch_size, file, category_tag, redirect_tag, first_n_lines, pool):
    logging.info('parse articles start')
    lines = list(smart_open.open(file, 'r'))
    if first_n_lines > 0:
        lines = lines[:first_n_lines]
    pool.map(partial(parse_articles_callback, batch_size, category_tag, redirect_tag), lines)
    logging.info('finish')

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("json_articles_file", help="path to json articles file", type=str)
    parser.add_argument("polimorfologik_file", help="path to polimorfologik  file", type=str)
    parser.add_argument("stop_words_file", help="path to stop words file", type=str)
    parser.add_argument("-c", "--category_tag", help="category tag", default="kategoria:", type=str)
    parser.add_argument("-r", "--redirect_tag", help="redirect tag", default="#PATRZ", type=str)
    parser.add_argument("-t", "--threads", help="threads", type=int, default=1, choices=range(1, 33), metavar="int")
    parser.add_argument("-b", "--batch_size", help="batch_size", type=int, default=10000, metavar="int")
    parser.add_argument('-f', '--first_n_lines', help="process only first n lines of each file", type=int, default=0)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)
    logging.info('start')
    logging.info('json: %s' % args.json_articles_file)
    logging.info('polimorfologik: %s' % args.polimorfologik_file)
    logging.info('stop_words: %s' % args.stop_words_file)
    logging.info('category_tag: %s' % args.category_tag)
    logging.info('redirect_tag: %s' % args.redirect_tag)
    logging.info('threads: %d' % args.threads)
    logging.info('batch size: %d' % args.batch_size)
    logging.info('first_n_lines: %d' % args.first_n_lines)

    pool = multiprocess.Pool(args.threads)
    category_tag = args.category_tag.strip().lower()
    redirect_tag = args.redirect_tag.strip().lower()
    first_n_lines = max(0, args.first_n_lines)
    preparse_polimorfologik(args.batch_size, args.polimorfologik_file, category_tag, first_n_lines)
    preparse_articles(args.batch_size, args.json_articles_file, category_tag, first_n_lines, pool)
    parse_articles(args.batch_size, args.json_articles_file, category_tag, redirect_tag, first_n_lines, pool)
    parse_stop_words(args.stop_words_file, first_n_lines)
