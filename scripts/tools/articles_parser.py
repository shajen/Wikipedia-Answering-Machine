from data.models import *

from collections import defaultdict
import mwparserfromhell
import operator
import re
import logging
import time

class ArticlesParser():
	def __init__ (self, batch_size, category_tag):
		self.batch_size = batch_size
		self.category_tag = category_tag

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

	def parseCategory(self, title, text, links):
		pass

	def parseRedirect(self, title, text, links, redirect_tag):
		try:
			if links:
				link = links[0].split('#')[0]
			else:
				link = text.replace(redirect_tag, '')
			link = link.replace('_', ' ').strip()
			redirected_to = Article.objects.get(title=link)
			Article.objects.filter(title=title).update(redirected_to=redirected_to)
		except Exception as e:
			logging.warning('exception during parsing redirect')
			logging.warning(e)
			logging.warning('title: %s, link: %s' % (title, link))

	def parseArticle(self, title, text, links):
		article = Article.objects.get(title=title)
		# for link in links:
		# 	if link.startswith(self.category_tag):
		# 		try:
		# 			linkedCategory, created = Category.objects.get_or_create(title=link[len(self.category_tag):])
		# 			article.categories.add(linkedCategory)
		# 		except:
		# 			#logging.warning('link category not found source article: %s, target category: %s' % (title, link))
		# 			pass
		# 	else:
		# 		try:
		# 			linkedArticle = Article.objects.get(title=link)
		# 			article.links.add(linkedArticle)
		# 		except:
		# 			#logging.warning('link article not found source article: %s, target article: %s' % (title, link))
		# 			pass
		title = self.normaliseText(title)
		text = self.normaliseText(text)
		probes = 10
		for i in range(probes):
			try:
				self.parseText(article, title, True)
				break
			except Exception as e:
				if i == probes - 1:
					logging.warning('exception during parsing title')
					logging.warning(e)
					logging.warning(title)
				time.sleep(1)
		for i in range(probes):
			try:
				self.parseText(article, text, False)
				break
			except Exception as e:
				if i == probes - 1:
					logging.warning('exception during parsing text')
					logging.warning(e)
					logging.warning(title)
				time.sleep(1)

	def parseText(self, article, text, isTitle):
		new_words = []
		words = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', text)
		word_to_id = {}
		for word in Word.objects.filter(changed_form__in=set(words)).values('id', 'changed_form'):
			word_to_id[word['changed_form']] = word['id']

		found_words = word_to_id.keys()
		for w in set(words):
			if len(w) > 1 and w not in found_words:
				new_words.append(Word(base_form=w, changed_form=w))
		Word.objects.bulk_create(new_words, ignore_conflicts=True, batch_size=self.batch_size)

		for word in Word.objects.filter(changed_form__in=(set(words) - found_words)).values('id', 'changed_form'):
			word_to_id[word['changed_form']] = word['id']

		positions = defaultdict(set)
		currentPos = 0
		for w in words:
			if len(w) > 1:
				currentPos += 1
				try:
					positions[word_to_id[w]].add(currentPos)
				except Exception as e:
					logging.warning('exception during searching word')
					logging.warning(e)
					logging.warning('word id: %s, article title: %s' % (w, article.title))

		new_occurrences = []
		for word_id in positions:
			p = ','.join(str(s) for s in positions[word_id])
			if len(p) > 2048:
				p = re.sub('\d+$', '', p[:2048])
				logging.warning('exception during insert Occurence')
				logging.warning('positions too long, truncated positions')
				logging.warning('word id: %s, article title: %s' % (word_id, article.title))
			new_occurrences.append(Occurrence(article=article, word_id=word_id, positions=p, positions_count=len(positions[word_id]), is_title=isTitle))

		Occurrence.objects.bulk_create(new_occurrences, ignore_conflicts=True, batch_size=self.batch_size)

	def addBaseForms(self, baseText, changedText):
		words = []
		baseText = re.sub('(\(.+?\))', ' ', baseText.lower())
		changedText = re.sub('(\(.+?\))', ' ', changedText.lower())
		baseWords = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', baseText)
		changedWords = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', changedText)
		if len(baseWords) == len(changedWords):
			if all((baseWords[i].isalpha() and changedWords[i].isalpha() and baseWords[i][:3] == changedWords[i][:3]) for i in range(len(baseWords))):
				for i in range(len(baseWords)):
					if baseWords[i] != changedWords[i]:
						words.append(Word(base_form=baseWords[i], changed_form=changedWords[i]))
		return words
