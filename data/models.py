from django.db import models
from django.core.validators import int_list_validator
from more_itertools import unique_everseen
import numpy as np
import statistics
import re
from collections import defaultdict

class ListField(models.TextField):
    def __init__(self, *args, **kwargs):
        self.token = kwargs.pop('token', ',')
        super().__init__(*args, **kwargs)

    def deconstruct(self):
        name, path, args, kwargs = super().deconstruct()
        kwargs['token'] = self.token
        return name, path, args, kwargs

    def to_python(self, value):
        class SubList(list):
            def __init__(self, token, *args):
                self.token = token
                super().__init__(*args)

            def __str__(self):
                return self.token.join(self)

        if isinstance(value, list):
            return value
        if value is None:
            return SubList(self.token)
        return SubList(self.token, value.split(self.token))

    def from_db_value(self, value, expression, connection):
        return self.to_python(value)

    def get_prep_value(self, value):
        if not value:
            return
        assert(isinstance(value, Iterable))
        return self.token.join(value)

    def value_to_string(self, obj):
        value = self.value_from_object(obj)
        return self.get_prep_value(value)

class Category(models.Model):
    title = models.CharField(max_length=255, unique=True)
    added_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name_plural = "Categories"

    def __str__(self):
        return self.title

class Article(models.Model):
    title = models.CharField(max_length=255, unique=True)
    links = models.ManyToManyField(
        to='self',
        related_name='links_relationship',
        symmetrical=False
    )
    redirected_to = models.ForeignKey(
        'self',
        related_name='redirection_relationship',
        null=True,
        on_delete=models.CASCADE
    )
    categories = models.ManyToManyField(Category)
    content_words = models.TextField(default='')
    content_words_count = models.PositiveIntegerField(default=0)
    title_words = models.TextField(default='')
    title_words_count = models.PositiveIntegerField(default=0)
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    def get_words(self, is_title, stop_words, top_words):
        if is_title:
            words = self.title_words.split(',')
        else:
            words = self.content_words.split(',')
        words = list(filter(lambda w: w != '', words))
        words = list(map(lambda w: int(w), words))
        words = list(filter(lambda w: w not in stop_words, words))
        words = np.array(words[:top_words], dtype=np.uint32)
        words = np.concatenate((words, np.zeros(shape=(top_words - words.shape[0]), dtype=np.uint32)), axis=0)
        return words

class Method(models.Model):
    name = models.CharField(max_length=255, unique=True)
    added_date = models.DateTimeField(auto_now_add=True)
    is_enabled = models.BooleanField(default=True, db_index=True)
    is_smaller_first = models.BooleanField()

    def __str__(self):
        return self.name

    def scores(self, questions_id=[], top_n=[1,10,100]):
        def p_score(n, positions):
            try:
                return len([p for p in positions if p<=n]) / len(positions)
            except ZeroDivisionError:
                return 0.0

        def mrr(positions):
            try:
                return 1 / statistics.harmonic_mean(positions)
            except statistics.StatisticsError:
                return 0.0

        if questions_id:
            answers = Answer.objects.filter(question_id__in=questions_id).values_list('id', flat=True)
        else:
            answers = Answer.objects.values_list('id', flat=True)

        positions = list(Solution.objects.filter(answer_id__in=answers, method_id=self.id).values_list('position', flat=True))
        p_scores = [p_score(n, positions) for n in top_n]
        return (len(positions), p_scores + [mrr(positions)])

    def preety_name(self):
        vector_match = re.search(r'type: tfi, title: (\d+), ngram: (\d+), mwiw: 0.00, n: (\d+), m: (\w+)', self.name)
        tfidf_match = re.search(r'type: tfi, title: (\d+), ngram: (\d+), mwiw: 0.00, n: (\d+), pf:', self.name)
        w2v_match = re.search(r'type: w2v, topn: (\d+), title: (\d), q: \d+, at: \d+, ac: \d+', self.name)
        if vector_match:
            title = int(vector_match.group(1))
            ngram = int(vector_match.group(2))
            n = int(vector_match.group(3))
            type = vector_match.group(4)
            if type == 'cosine':
                type = 'cosinusowa'
            return 'kontekstowa wektorowa miara %s\ndla n = %d, dane: %s %s artykułów' % (type, n, 'słowa' if ngram == 1 else 'bigramy', 'tytułów' if title == 1 else 'treści')
        elif tfidf_match:
            title = int(tfidf_match.group(1))
            ngram = int(tfidf_match.group(2))
            n = int(tfidf_match.group(3))
            if n == 0:
                return 'miara TF-IDF\ndane: %s %s artykułów' % ('słowa' if ngram == 1 else 'bigramy', 'tytułów' if title == 1 else 'treści')
            else:
                return 'kontekstowa miara TF-IDF dla n = %d\ndane: %s %s artykułów' % (n, 'słowa' if ngram == 1 else 'bigramy', 'tytułów' if title == 1 else 'treści')
        elif w2v_match:
            topn = int(w2v_match.group(1))
            title = int(w2v_match.group(2))
            if topn == 0:
                type = 'tylko słowa zawarte w pytaniu'
            elif topn == 10:
                type = '10 najbliższych słów'
            else:
                type = 'wszystkie słowa'
            return 'word2vec %s\ndane: %s artykułów' % (type, 'tytuły' if title == 1 else 'treści')
        elif 'type: cnn' in self.name:
            return 'CNN'
        elif 'type: dan' in self.name:
            return 'DAN'
        elif 'type: ean' in self.name:
            return 'algorytm ewolucyjny'
        return self.name

class Question(models.Model):
    name = models.TextField(max_length=1024)
    words = models.TextField(default='')
    words_count = models.PositiveIntegerField(default=0)
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    def corrected_articles(self):
        return list(map(lambda answer: answer.article, self.answer_set.all()))

    def corrected_articles_id(self):
        return list(map(lambda article: article.id, self.corrected_articles()))

    def get_words(self, stop_words, top_words):
        words = self.words.split(',')
        words = list(filter(lambda w: w != '', words))
        words = list(map(lambda w: int(w), words))
        words = list(filter(lambda w: w not in stop_words, words))
        words = np.array(words[:top_words], dtype=np.uint32)
        words = np.concatenate((words, np.zeros(shape=(top_words - words.shape[0]), dtype=np.uint32)), axis=0)
        return words

    def get_methods_results(self, methods):
        default_row = list(map(lambda m: float('Inf') if m.is_smaller_first else -float('Inf'), methods))
        data = defaultdict(lambda: default_row.copy())
        for i in range(len(methods)):
            for (article_id, weight) in Rate.objects.filter(question_id=self.id).filter(method_id=methods[i].id).values_list('article_id', 'weight'):
                data[article_id][i] = weight
        return data

class Answer(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return '%s, %s' % (self.article, self.question)

    class Meta:
        unique_together = ('article', 'question')

class Solution(models.Model):
    answer = models.ForeignKey(Answer, on_delete=models.CASCADE)
    method = models.ForeignKey(Method, on_delete=models.CASCADE)
    position = models.IntegerField()
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return '%s (%s: %d)' %(self.answer, self.method, self.position)

    class Meta:
        unique_together = ('answer', 'method')

class Word(models.Model):
    value = models.CharField(max_length=100, unique=True, db_index=True)
    is_stop_word = models.BooleanField(default=False, db_index=True)
    added_date = models.DateTimeField(auto_now_add=True)

    base_forms = models.ManyToManyField(
        to='self',
        related_name='changed_forms',
        symmetrical=False,
        through='WordForm'
    )

    def __str__(self):
        return self.value

class WordForm(models.Model):
    changed_word = models.ForeignKey(Word, on_delete=models.CASCADE, related_name='from_word')
    base_word = models.ForeignKey(Word, on_delete=models.CASCADE, related_name='to_word')
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return '%s - %s' % (self.base_form, self.changed_form)

    class Meta:
        unique_together = ('changed_word', 'base_word')

class ArticleOccurrence(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    word = models.ForeignKey(Word, on_delete=models.CASCADE)
    positions = models.CharField(max_length=20480)
    positions_count = models.PositiveSmallIntegerField(db_index=True)
    is_title = models.BooleanField(default=False, db_index=True)

    def __str__(self):
        return '%s - %s: %s' % (self.word, self.article, self.positions)

    class Meta:
        unique_together = ('article', 'word', 'is_title')

class QuestionOccurrence(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    word = models.ForeignKey(Word, on_delete=models.CASCADE)
    positions = models.CharField(max_length=20480)
    positions_count = models.PositiveSmallIntegerField(db_index=True)

    def __str__(self):
        return '%s - %s: %s' % (self.word, self.question, self.positions)

    class Meta:
        unique_together = ('question', 'word')

class Rate(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    method = models.ForeignKey(Method, on_delete=models.CASCADE)
    weight = models.FloatField()
    added_date = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('question', 'article', 'method')
