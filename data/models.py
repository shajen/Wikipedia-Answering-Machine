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
    title = models.CharField(max_length=1024)
    links = models.ManyToManyField(
        to='self',
        related_name='links_relationship',
        symmetrical=False
    )

class Method(models.Model):
    name = models.CharField(max_length=1024)

class Question(models.Model):
    name = models.CharField(max_length=1024)

class Answer(models.Model):
    article = models.ForeignKey(Article)
    question = models.ForeignKey(Question)

class Solution(models.Model):
    answer = models.ForeignKey(Answer)
    method = models.ForeignKey(Article)
    position = models.IntegerField()

class Word(models.Model):
    name = models.CharField(max_length=100)
    is_stop_word = models.BooleanField(default=False)

class Occurrence(models.Model):
    article = models.ForeignKey(Article)
    word = models.ForeignKey(Word)
    positions = ListField()
