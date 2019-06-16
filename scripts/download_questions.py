import urllib.request
import lxml.html
import argparse
import sys
import os
import shlex

WIKI_PAGE = "https://pl.wikipedia.org"
QUESTIONS_PAGE = WIKI_PAGE + "/wiki/Wikipedia:Wikireadery/Czy_wiesz,_%C5%BCe"

sys.path.append(os.path.dirname(__file__))

import tools.logger

def getSource(url):
    code = urllib.request.urlopen(url).read()
    return lxml.html.fromstring(code)

def getTitle(articleUrl):
    html = getSource(articleUrl)
    return html.xpath('//h1[@id="firstHeading"]')[0].text_content()

def processPage(questionUrl):
    html = getSource(questionUrl)
    for q in html.xpath('//div[@id="mw-content-text"]//p'):
        print(q.text_content().strip())
        for link in q.xpath('./b//a'):
            print(getTitle(WIKI_PAGE + link.attrib['href']))
        print('')

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0)
    args = parser.parse_args(args)

    tools.logger.configLogger(args.verbose)
    processPage(QUESTIONS_PAGE)
