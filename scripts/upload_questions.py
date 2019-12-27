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
import tools.questions_parser

def update_questions_words():
    logging.info('start update_questions_words')
    questions = []
    for question in Question.objects.all():
        logging.info('processing question: %d' % question.id)
        words = []
        for (word_id, positions) in QuestionOccurrence.objects.filter(question_id=question.id).values_list('word_id', 'positions').iterator():
            for position in positions.split(','):
                try:
                    words.append((int(position), word_id))
                except:
                    pass
        question.words_count = len(words)
        question.words = ','.join(list(map(lambda x: str(x[1]), sorted(words))))
        questions.append(question)
    Question.objects.bulk_update(questions, ['words', 'words_count'])

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", help="path to questions file")
    parser.add_argument("-m", "--min_article_character", help="min article character", type=int, default=200, metavar="int")
    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)
    questionsParser = tools.questions_parser.QuestionsParser(args.min_article_character)
    questionsParser.parse_file(args.questions)
    update_questions_words()
