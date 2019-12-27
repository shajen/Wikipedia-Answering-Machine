from collections import defaultdict
from data.models import *
from django import db
from django.db.models import Sum
from functools import reduce, partial
import argparse
import json
import logging
import multiprocessing
import numpy as np
import os
import re
import shlex
import smart_open
import sys

sys.path.append(os.path.dirname(__file__))

import tools.articles_parser
import tools.logger

def insert_objects(object_type, all_objects, batch_size):
    logging.info('inserting %d %s' % (len(all_objects), object_type.__name__))
    i = 0
    for objects in np.array_split(all_objects, max(1, len(all_objects) / batch_size)):
        i += batch_size
        try:
            logging.info('iteration #%d' % i)
            objects = list(objects)
            object_type.objects.bulk_create(objects, ignore_conflicts=True)
        except Exception as e:
            logging.warning('exception during insert %s:' % object_type.__name__)
            logging.warning(e)
            logging.warning(objects)
    logging.info('finish')

def insert_base_forms(base_forms, batch_size):
    words = set()
    for (base_form, changed_form) in base_forms:
        words.add(base_form)
        words.add(changed_form)
    words_objects = []
    for word in words:
        words_objects.append(Word(value=word))
    insert_objects(Word, words_objects, batch_size)

    word_value_to_id = {}
    for word in Word.objects.filter(value__in=words).values('value', 'id'):
        word_value_to_id[word['value']] = word['id']

    word_forms_objects = []
    for (base_form, changed_form) in base_forms:
        try:
            word_forms_objects.append(WordForm(base_word_id=word_value_to_id[base_form], changed_word_id=word_value_to_id[changed_form]))
        except Exception as e:
            logging.warning('exception durig insert_base_forms:')
            logging.warning(e)
            logging.warning('base_form: %s, changed_word: %s' % (base_form, changed_form))
    insert_objects(WordForm, word_forms_objects, batch_size)

def preparse_article_callback(batch_size, category_tag, line):
    try:
        articlesParser = tools.articles_parser.ArticlesParser(batch_size, category_tag)
        base_forms = []
        data = json.loads(line)
        title = data['title'].strip().lower()
        logging.debug('%s' % (title))

        for baseText in data['interlinks']:
            logging.debug('%s - %s' % (baseText, data['interlinks'][baseText]))
            base_forms.extend(articlesParser.addBaseForms(baseText, data['interlinks'][baseText]))

        if title.startswith(category_tag):
            title = title[len(category_tag):]
            return (None, Category(title=title), base_forms)
        else:
            return (Article(title=title), None, base_forms)
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
    db.connections.close_all()
    data = pool.map(partial(preparse_article_callback, batch_size, category_tag), lines)

    base_forms = [word for (a, c, base_forms) in data for word in base_forms]
    insert_base_forms(base_forms, batch_size)
    articles = [a for (a, c, w) in data if a is not None]
    insert_objects(Article, articles, batch_size)
    categories = [c for (a, c, w) in data if c is not None]
    insert_objects(Category, categories, batch_size)
    logging.info('finish')

def preparse_polimorfologik(batch_size, file, category_tag, first_n_lines):
    logging.info('preparse polimorfologik start')
    articlesParser = tools.articles_parser.ArticlesParser(batch_size, category_tag)
    lines = list(open(file, 'r'))
    if first_n_lines > 0:
        lines = lines[:first_n_lines]
    lines = list(set([tuple(re.split("[\t;]", line.strip().lower())[:2]) for line in lines]))
    logging.info('inserting %d words\n' % len(lines))
    base_forms = []
    for (changed_form, base_form) in lines:
        base_forms.append((base_form, changed_form))
    insert_base_forms(base_forms, batch_size)
    logging.info('finish')

def parse_stop_words(file, first_n_lines):
    logging.info('parse stop words start')
    lines = list(open(file, 'r'))
    if first_n_lines > 0:
        lines = lines[:first_n_lines]
    stop_words = [line.strip().lower() for line in lines]
    Word.objects.filter(value__in=stop_words).update(is_stop_word=True)
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
            articlesParser.parseRedirect(title, text, links, redirect_tag)
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
    db.connections.close_all()
    pool.map(partial(parse_articles_callback, batch_size, category_tag, redirect_tag), lines)
    logging.info('finish')

def update_articles_words_count(is_title):
    logging.info('start update_articles_words_count is_title: %d' % is_title)
    logging.info('reading stop words')
    stop_words = list(map(lambda x: x['id'], Word.objects.filter(is_stop_word = True).values('id')))
    articles = ArticleOccurrence.objects.values('article_id').filter(is_title=is_title).exclude(word_id__in=stop_words).annotate(count=Sum('positions_count'))
    logging.info('reading articles words count')
    i = 0
    for article in articles:
        i += 1
        if i % 10000 == 0:
            logging.info(article)
            logging.info('iteration #%d' % i)
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

def update_articles_words(is_title, chunk_size):
    logging.info('start update_articles_words is_title: %d' % is_title)
    current = 0
    total = Article.objects.count() / chunk_size

    def update(articles):
        if is_title:
            Article.objects.bulk_update(articles, ['title_words'])
        else:
            Article.objects.bulk_update(articles, ['content_words'])

    articles = []
    for article in Article.objects.order_by('id').iterator():
        words = []
        for (word_id, positions) in ArticleOccurrence.objects.filter(article_id=article.id, is_title=is_title).values_list('word_id', 'positions').iterator():
            for position in positions.split(','):
                try:
                    words.append((int(position), word_id))
                except:
                    pass
        words = ','.join(list(map(lambda x: str(x[1]), sorted(words))))
        if is_title:
            article.title_words = words
        else:
            article.content_words = words
        articles.append(article)
        if len(articles) >= chunk_size:
            update(articles)
            articles = []
            current += 1
            logging.debug("progress: %d/%d (%.2f %%)" % (current, total, current / total * 100))
    update(articles)

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

    pool = multiprocessing.Pool(args.threads)
    category_tag = args.category_tag.strip().lower()
    redirect_tag = args.redirect_tag.strip().lower()
    first_n_lines = max(0, args.first_n_lines)
    preparse_polimorfologik(args.batch_size, args.polimorfologik_file, category_tag, first_n_lines)
    preparse_articles(args.batch_size, args.json_articles_file, category_tag, first_n_lines, pool)
    parse_articles(args.batch_size, args.json_articles_file, category_tag, redirect_tag, first_n_lines, pool)
    parse_stop_words(args.stop_words_file, first_n_lines)
    update_articles_words(True, 1000)
    update_articles_words_count(True)
    update_articles_words(False, 1000)
    update_articles_words_count(False)
