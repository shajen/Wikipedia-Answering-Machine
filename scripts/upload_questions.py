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

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", help="path to questions file")
    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)
    questionsParser = tools.questions_parser.QuestionsParser()
    questionsParser.parse_file(args.questions)
