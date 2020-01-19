from django.contrib import admin
from data.models import *

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'title', 'added_date')
	search_fields = ('title', )

@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'title', 'redirected_to', 'title_words_count', 'content_words_count', 'added_date')
	list_select_related = ('redirected_to',)
	raw_id_fields = ('links', 'redirected_to', 'categories')
	search_fields = ('title', )

@admin.register(Method)
class MethodAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'name', 'is_enabled', 'is_smaller_first', 'added_date')
	search_fields = ('name', )

@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'name', 'words_count', 'added_date')
	search_fields = ('name', )

@admin.register(Answer)
class AnswerAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'article', 'question', 'added_date')
	list_select_related = ('article', 'question',)
	raw_id_fields = ('article', 'question')

@admin.register(Solution)
class SolutionAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'answer', 'method', 'position', 'added_date')
	list_select_related = ('answer', )
	raw_id_fields = ('method', 'answer')
	search_fields = ('answer__article__title', 'method__name')

@admin.register(Word)
class WordAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'value', 'is_stop_word', 'added_date')
	search_fields = ('value', )

@admin.register(WordForm)
class WordFormAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'changed_word', 'base_word', 'added_date')
	list_select_related = ('changed_word', 'base_word',)
	raw_id_fields = ('changed_word', 'base_word')
	search_fields = ('changed_word__value', 'base_word__value',)

@admin.register(ArticleOccurrence)
class ArticleOccurrenceAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'article_id', 'word_id', 'positions', 'positions_count', 'is_title')
	raw_id_fields  = ('article', 'word')
	show_full_result_count = False

@admin.register(QuestionOccurrence)
class QuestionOccurrenceAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'question_id', 'word_id', 'positions', 'positions_count')
	raw_id_fields  = ('question', 'word')
	show_full_result_count = False

@admin.register(Rate)
class RateAdmin(admin.ModelAdmin):
	list_per_page = 100
	list_display = ('id', 'question_id', 'article_id', 'method_id', 'weight', 'added_date')
	raw_id_fields = ('question', 'article', 'method')
	show_full_result_count = False
