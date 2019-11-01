from data.models import *
from django.db.models import Value
from django.db.models.functions import Replace

import shlex
import argparse

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help="source pattern string", type=str, default='', metavar="source")
    parser.add_argument('destination', help="destination pattern string", type=str, default='', metavar="destination")
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args(args)

    Method.objects.update(name=Replace('name', Value(args.source), Value(args.destination)))
