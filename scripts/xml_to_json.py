#!/usr/bin/python

import gensim
import gensim.scripts.segment_wiki
import logging
import os
import sys
import shlex
import argparse

sys.path.append(os.path.dirname(__file__))

import tools.logger

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("wiki_articles_xml_bz2_file", help="path to wiki xml bz2 articles file", type=str)
    parser.add_argument("json_articles_file", help="path to json articles file", type=str)
    parser.add_argument("-t", "--threads", help="threads", type=int, default=1, choices=range(1, 33), metavar="int")
    parser.add_argument("-m", "--min_article_character", help="min article character", type=int, default=200, metavar="int")
    parser.add_argument("-i", "--interlinks", help="user interlinks", action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)
    logging.info('start')
    logging.info('threads: %d' % args.threads)

    gensim.corpora.wikicorpus.IGNORED_NAMESPACES.remove('Category')
    logging.info('ignored namespaces: %s' % gensim.corpora.wikicorpus.IGNORED_NAMESPACES)

    gensim.scripts.segment_wiki.segment_and_write_all_articles(args.wiki_articles_xml_bz2_file, args.json_articles_file, args.min_article_character, args.threads, args.interlinks)
    logging.info('finish')
