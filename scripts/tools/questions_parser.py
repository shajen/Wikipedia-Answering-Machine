from data.models import *

from collections import defaultdict
import logging
import re

class QuestionsParser:
    def __init__(self, min_article_character):
        logging.info('reading articles')
        self.min_article_character = min_article_character
        self.articles_redirected_to = {}
        self.articles_content_count = {}
        self.articles_title = {}
        self.titles_article = {}
        for (id, content_words_count, title, redirected_to_id) in Article.objects.values_list('id', 'content_words_count', 'title', 'redirected_to_id'):
            if redirected_to_id:
                self.articles_redirected_to[id] = redirected_to_id
            self.articles_content_count[id] = content_words_count
            self.articles_title[id] = title
            self.titles_article[title] = id

    def parse_file(self, file):
        questions_id = []
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
            questions_id.append(self.parse_question(question, answers))
        logging.info('finish')
        return questions_id

    def parse_question(self, question_text, answers):
        logging.debug('question:')
        logging.debug(question_text)
        logging.debug('answers:')
        logging.debug(answers)

        if answers:
            try:
                question, created = Question.objects.get_or_create(name=question_text)
                words = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', question_text)
                words_objects = []
                for word in words:
                    words_objects.append(Word(value=word))
                Word.objects.bulk_create(words_objects, ignore_conflicts=True)

                word_value_to_id = {}
                for word in Word.objects.filter(value__in=words).values('id', 'value'):
                    word_value_to_id[word['value']] = word['id']

                words_positions = defaultdict(list)
                position = 0
                for word in words:
                    position += 1
                    words_positions[word].append(position)

                question_occurrence_objects = []
                for word in set(words):
                    try:
                        positions = ','.join(str(s) for s in words_positions[word])
                        question_occurrence_objects.append(QuestionOccurrence(question=question, word_id=word_value_to_id[word], positions=positions, positions_count=len(words_positions[word])))
                    except Exception as e:
                        logging.warning('exception during searching word id:')
                        logging.warning(e)
                        logging.warning('word: %s' % (word))
                QuestionOccurrence.objects.bulk_create(question_occurrence_objects, ignore_conflicts=True)

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
                    if self.articles_content_count[article_id] >= self.min_article_character:
                        Answer.objects.create(question=question, article_id=article_id)
                except:
                    pass
            try:
                if question.answer_set.count() == 0:
                    logging.debug('delete question without answer: %d %s' % (question.id, question_text))
                    question.delete()
                else:
                    return question.id
            except Exception as e:
                logging.warning('exception during delete question:')
                logging.warning(e)
                logging.warning(question_text)
        return -1
