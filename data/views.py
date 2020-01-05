from data.models import *
from django.core.exceptions import ObjectDoesNotExist
from django.core.paginator import Paginator
from django.http import HttpResponse, Http404
from django.shortcuts import render
from django.template import loader

__OBJECTS_PER_PAGE = 10000

def index(request):
    return render(request, 'index.html')

def article(request, id):
    try:
        article = Article.objects.get(id=id)
        answers = article.answer_set.select_related('question')
        return render(request, 'data/article.html', {'article' :  article, 'answers' : answers})
    except ObjectDoesNotExist:
        raise Http404('Article %d not found' % id)

def articles(request):
    order_by = request.GET.get('order_by', 'title')
    page = request.GET.get('page')
    objects_per_page = request.GET.get('objects_per_page', __OBJECTS_PER_PAGE)
    paginator = Paginator(Article.objects.filter(redirected_to=None, content_words_count__gte=1, title_words_count__gte=1).order_by(order_by).all(), objects_per_page)
    return render(request, 'data/articles.html', {'objects': paginator.get_page(page), 'total_count': paginator.count, 'order_by' : order_by })

def question(request, id):
    try:
        question = Question.objects.get(id=id)
        answers = question.answer_set.select_related('article')
        return render(request, 'data/question.html', {'question' :  question, 'answers' : answers})
    except ObjectDoesNotExist:
       raise Http404('Question %d not found' % id)

def questions(request):
    order_by = request.GET.get('order_by', 'name')
    page = request.GET.get('page')
    objects_per_page = request.GET.get('objects_per_page', __OBJECTS_PER_PAGE)
    paginator = Paginator(Question.objects.order_by(order_by).all(), objects_per_page)
    return render(request, 'data/questions.html', {'objects': paginator.get_page(page), 'total_count': paginator.count, 'order_by' : order_by })

def method(request, id):
    try:
        return render(request, 'data/method.html', {'method' : Method.objects.get(id=id) })
    except ObjectDoesNotExist:
        raise Http404('Method %d not found' % id)

def methods(request):
    order_by = request.GET.get('order_by', 'name')
    page = request.GET.get('page')
    objects_per_page = request.GET.get('objects_per_page', __OBJECTS_PER_PAGE)
    paginator = Paginator(Method.objects.filter(is_enabled=True).order_by(order_by).all(), objects_per_page)
    return render(request, 'data/methods.html', {'objects': paginator.get_page(page), 'total_count': paginator.count, 'order_by' : order_by })
