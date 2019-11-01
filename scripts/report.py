from data.models import *

import argparse
import logging
import os
import shlex
import sys

sys.path.append(os.path.dirname(__file__))

import tools.logger
import tools.report_manager

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--top_answers", help="count selected top answers", type=str, default='1,10,100', metavar="n1,n2,n3...")
    parser.add_argument("-mp", "--method_patterns", help="print only method with patterns", type=str, default='', metavar="pattern")
    parser.add_argument("-qap", "--question_answer_position", help="print questions start from answer position", type=int, default=100, metavar='position')
    parser.add_argument("-qab", "--question_better", help="print questions answers from better to worst way", action='store_true')
    parser.add_argument("-qc", "--question_count", help="print n questions", type=int, default=3, metavar='n')
    parser.add_argument("-hs", "--hide_score", help="hide score below treshlod", type=float, default=0.0, metavar='f')
    parser.add_argument("-snf", "--show_not_found", help="show not found answers", action='store_true')
    parser.add_argument("-nc", "--no_color", help="switch off color", action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)

    tops = [int(x) for x in args.top_answers.split(',')]
    qam = tools.report_manager.ReportManager(args.no_color)
    data = {
        'tops' : tops,
        'methodPatterns' : args.method_patterns,
        'questionAnswerPosition' : args.question_answer_position,
        'questionBetter' : args.question_better,
        'questionCount' : args.question_count,
        'showNotFound' : args.show_not_found,
        'hideScoreTreshold' : args.hide_score,
        }
    qam.process(data)
