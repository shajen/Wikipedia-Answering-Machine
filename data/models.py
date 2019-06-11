from django.db import models
from django.core.validators import int_list_validator

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

class Article(models.Model):
    title = models.CharField(max_length=255, unique=True)
    links = models.ManyToManyField(
        to='self',
        related_name='links_relationship',
        symmetrical=False
    )
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class Method(models.Model):
    name = models.CharField(max_length=255, unique=True)
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class Question(models.Model):
    name = models.CharField(max_length=255, unique=True)
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

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
    method = models.ForeignKey(Article, on_delete=models.CASCADE)
    position = models.IntegerField()
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return '%s (%s: %d)' %(self.answer, self.method, self.position)

    class Meta:
        unique_together = ('answer', 'method')

class Word(models.Model):
    base_form = models.CharField(max_length=100)
    changed_form = models.CharField(max_length=100, unique=True)
    is_stop_word = models.BooleanField(default=False)
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return '%s (%s)' % (self.changed_form, self.base_form)

class Occurrence(models.Model):
    article = models.ForeignKey(Article, on_delete=models.CASCADE)
    word = models.ForeignKey(Word, on_delete=models.CASCADE)
    positions = models.CharField(max_length=2048)
    positions_count = models.PositiveSmallIntegerField()
    is_title = models.BooleanField(default=False)

    def __str__(self):
        return '%s - %s: %s' % (self.word, self.article, self.positions)

    class Meta:
        unique_together = ('article', 'word', 'is_title')
