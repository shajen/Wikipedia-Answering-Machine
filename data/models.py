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
    content_words_count = models.PositiveIntegerField(default=0)
    title_words_count = models.PositiveIntegerField(default=0)
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

class Method(models.Model):
    name = models.CharField(max_length=255, unique=True)
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

class Question(models.Model):
    name = models.TextField(max_length=1024)
    added_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    def get_ngrams(self, ngram):
        occurrences = []
        for occurrence in self.questionoccurrence_set.all().values('word', 'positions'):
            for position in map(lambda x: int(x), occurrence['positions'].split(',')):
                occurrences.append((occurrence['word'], position))
        occurrences = sorted(occurrences, key=lambda occurrence: occurrence[1])

        words = list(map(lambda occurrence: occurrence[0], occurrences))
        words = Word.objects.filter(id__in=words, is_stop_word=False).values_list('id', flat=True)
        occurrences = list(filter(lambda occurrence: occurrence[0] in words, occurrences))
        
        words = []
        for i in range(0, len(occurrences) - ngram + 1):
            current_occurrences = occurrences[i:i+ngram]
            if current_occurrences[-1][1] - current_occurrences[0][1] == ngram - 1:
                word = list(map(lambda occurrence: occurrence[0], current_occurrences))
                words.append(tuple(word))
        return words

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
