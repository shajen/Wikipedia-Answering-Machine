from data.models import *

import argparse
import json
import logging
import os
import shlex
import re

import sys
sys.path.append(os.path.dirname(__file__))

import tools.logger

def run(*args):
    logging.info('start')
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", help="path to questions file")
    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)

    fp = open(args.questions, 'r')
    while True:
        query = fp.readline()
        query = re.sub('[^\w ]+',' ', query)
        query = re.sub(' +',' ', query).lower().strip()
        if query == '':
            break
        logging.debug('read query:')
        logging.debug(query)
        logging.debug('answers:')
        answers = []
        answer = fp.readline().strip()
        while answer != '':
            try:
                article = Article.objects.get(title=answer.strip().lower())
                logging.debug(article.title)
                answers.append(article)
            except Exception as e:
                pass
            answer = fp.readline().strip()
        if answers:
            try:
                question, created = Question.objects.get_or_create(name=query)
            except Exception as e:
                logging.warning('exception during get_or_create question:')
                logging.warning(e)
                logging.warning(query)
            for answer in answers:
                try:
                    Answer.objects.create(question=question, article=answer)
                except:
                    pass
    logging.info('finish')
