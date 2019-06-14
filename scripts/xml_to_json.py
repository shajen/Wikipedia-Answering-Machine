#!/usr/bin/python

import gensim
import gensim.scripts.segment_wiki
import logging

gensim.corpora.wikicorpus.IGNORED_NAMESPACES.remove('Category')
logging.info('ignored namespaces: %s' % gensim.corpora.wikicorpus.IGNORED_NAMESPACES)

in_file='/home/shajen/mgr/data/in/plwiki-20190601-pages-articles-multistream.xml.bz2'
out_file='/home/shajen/mgr/data/in/plwiki-20190601-pages-articles-multistream.json.gz'
min_article_character=5
thread=8
interlinks=True
gensim.scripts.segment_wiki.segment_and_write_all_articles(in_file, out_file, min_article_character, thread, interlinks)
#python -m gensim.scripts.segment_wiki -i -f plwiki-latest-pages-articles-multistream.xml.bz2 -o plwiki-latest-pages-articles-multistream.json.gz
