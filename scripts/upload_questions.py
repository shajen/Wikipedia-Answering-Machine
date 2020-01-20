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

def update_questions_words(questions=Question.objects.all()):
    logging.info('start update_questions_words')
    logging.info('count: %d' % len(questions))
    questions_to_update = []
    for question in questions:
        logging.debug('processing question: %d' % question.id)
        words = []
        for (word_id, positions) in QuestionOccurrence.objects.filter(question_id=question.id).values_list('word_id', 'positions').iterator():
            for position in positions.split(','):
                try:
                    words.append((int(position), word_id))
                except:
                    pass
        question.words_count = len(words)
        question.words = ','.join(list(map(lambda x: str(x[1]), sorted(words))))
        questions_to_update.append(question)
    Question.objects.bulk_update(questions_to_update, ['words', 'words_count'])

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("-qf", "--questions_file", help="path to questions file", type=str)
    parser.add_argument("-q", "--questions", help="question to add", type=str)
    parser.add_argument("-a", "--answer", help="answer for question", type=str)
    parser.add_argument("-m", "--min_article_character", help="min article character", type=int, default=200, metavar="int")
    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)
    questionsParser = tools.questions_parser.QuestionsParser(args.min_article_character)
    if args.questions_file:
        logging.info('parse file: %s' % args.questions_file)
        questions_id = questionsParser.parse_file(args.questions_file)
        update_questions_words(Question.objects.filter(id__in=questions_id))
    if args.questions and args.answer:
        questions_id = [questionsParser.parse_question(question, [args.answer]) for question in args.questions.split(';')]
        update_questions_words(Question.objects.filter(id__in=questions_id))
    logging.info("update questions")
