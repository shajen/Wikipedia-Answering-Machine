from data.models import *

import logging
import re

class QuestionsParser:
    def __init__(self):
        logging.info('reading articles')
        articles = Article.objects.values('id', 'content_words_count', 'title', 'redirected_to_id')
        self.articles_redirected_to = {}
        self.articles_content_count = {}
        self.articles_title = {}
        self.titles_article = {}
        for article in articles:
            if article['redirected_to_id']:
                self.articles_redirected_to[article['id']] = article['redirected_to_id']
            self.articles_content_count[article['id']] = article['content_words_count']
            self.articles_title[article['id']] = article['title']
            self.titles_article[article['title']] = article['id']

    def parse_file(self, file):
        logging.info('parsing questions: %s' % file)
        fp = open(file, 'r')
        while True:
            question = fp.readline()
            question = re.sub('[^\w ]+',' ', question)
            question = re.sub(' +',' ', question).lower().strip()
            if question == '':
                break
            answers = []
            answer = fp.readline().strip()
            while answer != '':
                answers.append(answer.lower())
                answer = fp.readline().strip()
            self.parse_question(question, answers)
        logging.info('finish')

    def parse_question(self, question_text, answers):
        logging.debug('question:')
        logging.debug(question_text)
        logging.debug('answers:')
        logging.debug(answers)

        if answers:
            try:
                question, created = Question.objects.get_or_create(name=question_text)
            except Exception as e:
                logging.warning('exception during get_or_create question:')
                logging.warning(e)
                logging.warning(question_text)
            for answer in answers:
                try:
                    article_id = self.titles_article[answer]
                    while article_id in self.articles_redirected_to and self.articles_redirected_to[article_id] != article_id:
                        logging.debug('%s redirected to %s' % (self.articles_title[article_id], self.articles_title[self.articles_redirected_to[article_id]]))
                        article_id = self.articles_redirected_to[article_id]
                    Answer.objects.create(question=question, article_id=article_id)
                except:
                    pass
