from data.models import *

from collections import defaultdict
import mwparserfromhell
import operator
import re

class ArticlesParser():
	def __init__ (self):
		pass

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
		article = Article.objects.filter(title__icontains=title)[0]
		for link in links:
			try:
				linkedArticle = Article.objects.filter(title__icontains=link)[0]
				article.links.add(linkedArticle)
			except:
				pass
		title = self.normaliseText(title)
		text = self.normaliseText(text)
		self.parseText(article, title, True)
		self.parseText(article, text, False)

	def parseText(self, article, text, isTitle):
		words = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', text)
		positions = defaultdict(set)
		currentPos = 0
		for w in words:
			if len(w) > 1:
				currentPos += 1
				if w != '.':
					try:
						word = Word.objects.filter(changed_form=w)[0]
					except:
						word, created = Word.objects.get_or_create(base_form=w, changed_form=w)
					positions[word.id].add(currentPos)

		for word_id in positions:
			p = ','.join(str(s) for s in positions[word_id])
			Occurrence.objects.create(article=article, word_id=word_id, positions=p, is_title=isTitle)

	def addBaseForms(self, baseText, changedText):
		baseText = re.sub('(\(.+?\))', ' ', baseText.lower())
		changedText = re.sub('(\(.+?\))', ' ', changedText.lower())
		baseWords = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', baseText)
		changedWords = re.findall('(\d+(?:\.|,)\d+|\w+|\.)', changedText)
		if len(baseWords) == len(changedWords):
			if all(baseWords[i][:3] == changedWords[i][:3] for i in range(len(baseWords))):
				for i in range(len(baseWords)):
					if baseWords[i] != changedWords[i]:
						Word.objects.get_or_create(base_form=baseWords[i], changed_form=changedWords[i])
