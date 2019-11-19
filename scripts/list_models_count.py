from data.models import *
from django.db.models import Sum

import shlex
import argparse

def list_model_count(model):
    print('%s %d' % (model.__name__, model.objects.count()))

def list_model_sum(model, sum_field):
    data = model.objects.aggregate(Sum(sum_field))
    print('%s %d' % (model.__name__, data['%s__sum' % sum_field]))

def list_model_field_count(model, field):
    print('%s %s %d' % (model.__name__, field, model.objects.values(field).distinct().count()))

def run(*args):
    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []

    list_model_count(Word)
    list_model_field_count(WordForm, 'base_word')
    list_model_field_count(WordForm, 'changed_word')
    list_model_count(Article)
    list_model_sum(ArticleOccurrence, 'positions_count')
    list_model_count(Question)
    list_model_sum(QuestionOccurrence, 'positions_count')
    list_model_count(Answer)
