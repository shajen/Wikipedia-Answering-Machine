from django.contrib import admin
from data.models import *

class ArticleAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'title')

class MethodAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'name')

class QuestionAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'name')

class AnswerAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'article', 'question')

class SolutionAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'answer', 'method', 'position')

class WordAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'value', 'is_stop_word')

class ArticleOccurrenceAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'word', 'article', 'positions', 'positions_count', 'is_title')

class QuestionOccurrenceAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'word', 'question', 'positions', 'positions_count')

admin.site.register(Article, ArticleAdmin)
admin.site.register(Method, MethodAdmin)
admin.site.register(Question, QuestionAdmin)
admin.site.register(Answer, AnswerAdmin)
admin.site.register(Solution, SolutionAdmin)
admin.site.register(Word, WordAdmin)
admin.site.register(ArticleOccurrence, ArticleOccurrenceAdmin)
admin.site.register(QuestionOccurrence, QuestionOccurrenceAdmin)
