from data.models import *
import numpy as np
import logging
from termcolor import colored

class ResultsPresenter():
    def __colored(text, colour):
        return colored(text, colour, attrs={'bold'})

    def __get_weight(weight, is_smaller_first):
        if np.isnan(weight):
            if is_smaller_first:
                return float(10**5)
            else:
                return 0.0
        else:
            return weight

    def __print(corrected_articles_id, position, articles_id, scores, distances, question, method, is_smaller_first):
        i = np.where(scores == position)[0][0]
        distance = distances[i]
        article = Article.objects.get(id=articles_id[i])
        colour = 'green' if articles_id[i] in corrected_articles_id else 'red'
        sign = '*' if articles_id[i] in corrected_articles_id else ' '
        logging.warning(ResultsPresenter.__colored(' %sposition: %6d, distance: %5.4f, article (%7d): %s' % (sign, position+1, distance, article.id, article), colour))
        return Rate(weight=ResultsPresenter.__get_weight(distance, is_smaller_first), article=article, question=question, method=method)

    def present(question, articles_id, distances, method, debug_top_items, is_smaller_first):
        logging.warning(ResultsPresenter.__colored('method (%d): %s' % (method.id, method.name), 'cyan'))
        rates = []
        if is_smaller_first:
            scores = np.argsort(np.argsort(distances))
        else:
            scores = np.argsort(np.argsort(distances)[::-1])
        corrected_articles_id = list(question.answer_set.all().values_list('article_id', flat=True))
        logging.warning(ResultsPresenter.__colored('question (%5d): %s' % (question.id, question), 'yellow'))
        for position in range(0, min(distances.shape[0], debug_top_items)):
            rates.append(ResultsPresenter.__print(corrected_articles_id, position, articles_id, scores, distances, question, method, is_smaller_first))

        for answer in question.answer_set.all():
            try:
                i = articles_id.index(answer.article_id)
                position = scores[i]
                if (position >= debug_top_items):
                    rates.append(ResultsPresenter.__print(corrected_articles_id, position, articles_id, scores, distances, question, method, is_smaller_first))
                Solution.objects.create(position=position+1, answer=answer, method=method)
            except:
                position = 10**5
                distance = np.nan
                logging.warning(ResultsPresenter.__colored(' %sposition: %6d, distance: %5.4f, article (%7d): %s' % ('*', position, distance, answer.article.id, answer.article), 'green'))
                Solution.objects.create(position=position, answer=answer, method=method)
                rates.append(Rate(weight=ResultsPresenter.__get_weight(distance, is_smaller_first), question=question, article=answer.article, method=method))
        Rate.objects.bulk_create(rates, ignore_conflicts=True)
        logging.info('')

    def is_already_solved(question, method_name):
        try:
            answers = Answer.objects.filter(question=question).values_list('id', flat=True)
            method = Method.objects.get(name=method_name)
            return Solution.objects.filter(method=method, answer_id__in=answers).count() == len(answers)
        except Exception as e:
            return False

    def log_article(article_id):
        if logging.getLogger().level != logging.DEBUG:
            return

        article = Article.objects.get(id=article_id)
        logging.debug('article: %7d, %s' % (article.id, article.title))

    def log_question(question_id):
        if logging.getLogger().level != logging.DEBUG:
            return

        question = Question.objects.get(id=question_id)
        logging.debug('question: %7d, %s' % (question.id, question.name))

    def log_words(words_id):
        if logging.getLogger().level != logging.DEBUG:
            return

        logging.debug('words count: %d' % len(words_id))
        for word in words_id:
            logging.debug('  word %7d: %s' % (word, Word.objects.get(id=word).value))
