from data.models import *

import argparse
import logging
import os
import re
import sys

sys.path.append(os.path.dirname(__file__))

import logger

if 'manage' in sys.argv[0]:
    sys.argv = sys.argv[2:]

parser = argparse.ArgumentParser()
parser.add_argument("questions", help="path to questions file", metavar="questions_file")
parser.add_argument('-v', '--verbose', action='count', default=0)
args = parser.parse_args()

logger.configLogger(args.verbose)

fp = open(args.questions, 'r')
while True:
    query = fp.readline()
    query = re.sub('[^\w ]+',' ', query)
    query = re.sub(' +',' ', query).lower().strip()
    if query == '':
        break
    logging.info('read query:')
    logging.info(query)
    logging.info('answers:')
    answers = []
    answer = fp.readline().strip()
    while answer != '':
        try:
            article = Article.objects.filter(title__icontains=answer)[0]
            answers.append(article)
        except:
            pass
        answer = fp.readline().strip()
    if answers:
        try:
            question = Question.objects.filter(name__icontains=query)[9]
            logging.warning('question already exist in database')
            logging.warning(question)
        except:
            question = Question.objects.create(name=query)
            logging.info('create in database')
            logging.info(question)
        for answer in answers:
            Answer.objects.create(question=question, article=answer)
