from data.models import *

from collections import defaultdict
import mwparserfromhell
import operator
import re
import logging
import time

class ArticlesParser():
	def __init__ (self, batch_size):
		self.batch_size = batch_size

	def removeObjects(self, text, objects):
		for t in objects:
			try:
				text.remove(t)
			except:
				pass

	def normaliseText(self, text):
		text = mwparserfromhell.parse(text)
		self.removeObjects(text, text.ifilter_arguments())
		self.removeObjects(text, text.ifilter_comments())
		self.removeObjects(text, text.ifilter_external_links())
		self.removeObjects(text, text.ifilter_headings())
		self.removeObjects(text, text.ifilter_html_entities())
		self.removeObjects(text, text.ifilter_templates())

		for t in text.ifilter_tags():
			try:
				text.replace(t, t.contents)
			except:
				pass

		for t in text.ifilter_wikilinks():
			try:
				text.replace(t, t.title)
			except:
				pass

		return text.strip_code().lower()

	def parseLinks(self, text):
		for t in mwparserfromhell.parse(text).ifilter_wikilinks():
			try:
				if t.text is not None:
					self.addBaseForms(str(t.title), str(t.text))
			except:
				pass

	def parseArticle(self, title, text, links):
		article = Article.objects.get(title=title)
		for link in links:
			try:
				category_tag = 'kategoria:'
				if link.startswith(category_tag):
					linkedCategory, created = Category.objects.get_or_create(title=link[len(category_tag):])
					article.categories.add(linkedCategory)
				else:
					linkedArticle = Article.objects.get(title=link)
					article.links.add(linkedArticle)
			except:
				#logging.warning('link article not found source article: %s, target article: %s' % (title, link))
				pass
		title = self.normaliseText(title)
		text = self.normaliseText(text)
		for i in range(10):
			try:
				self.parseText(article, title, True)
				if i != 0:
					logging.warning('success parsing title article: %s' % title)
				break
			except:
				logging.warning('retry parse title article: %s' % title)
				time.sleep(1)
		for i in range(10):
			try:
				self.parseText(article, text, False)
				if i != 0:
					logging.warning('success parsing text article: %s' % title)
				break
			except Exception as e:
				logging.warning('retry parse text article: %s' % title)
				time.sleep(1)

	def parseText(self, article, text, isTitle):
		new_words = []
		words = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', text)
		for w in set(words):
			if len(w) > 1:
				new_words.append(Word(base_form=w, changed_form=w))
		Word.objects.bulk_create(new_words, ignore_conflicts=True, batch_size=self.batch_size)
		created_words = Word.objects.filter(changed_form__in=set(words))

		positions = defaultdict(set)
		currentPos = 0
		for w in words:
			if len(w) > 1:
				currentPos += 1
				try:
					word = list(filter(lambda x: x.changed_form==w, created_words))[0]
					positions[word.id].add(currentPos)
				except:
					logging.warning('searching word id error:')
					logging.warning(w)

		new_occurrences = []
		for word_id in positions:
			p = ','.join(str(s) for s in positions[word_id])
			new_occurrences.append(Occurrence(article=article, word_id=word_id, positions=p, positions_count=len(positions[word_id]), is_title=isTitle))

		Occurrence.objects.bulk_create(new_occurrences, ignore_conflicts=True, batch_size=self.batch_size)

	def addBaseForms(self, baseText, changedText):
		words = []
		baseText = re.sub('(\(.+?\))', ' ', baseText.lower())
		changedText = re.sub('(\(.+?\))', ' ', changedText.lower())
		baseWords = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', baseText)
		changedWords = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', changedText)
		if len(baseWords) == len(changedWords):
			if all(baseWords[i][:3] == changedWords[i][:3] for i in range(len(baseWords))):
				for i in range(len(baseWords)):
					if baseWords[i] != changedWords[i]:
						words.append(Word(base_form=baseWords[i], changed_form=changedWords[i]))
		return words
