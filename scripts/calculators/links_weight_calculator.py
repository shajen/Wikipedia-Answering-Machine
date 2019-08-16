from collections import defaultdict
from data.models import *
import calculators.weight_calculator
import logging

class LinksWeightCalculator(calculators.weight_calculator.WeightCalculator):
    def __init__(self, debug_top_items):
        super().__init__(debug_top_items)
        logging.info('start reading links')
        self.articles_links = defaultdict(set)
        for data in Article.objects.values('id','links'):
            self.articles_links[data['id']].add(data['links'])
        logging.info('finish reading')

    def __count_articles_links_weight(self, articles_weight, n):
        articles_links_items_weight = defaultdict(defaultdict)
        articles_reverse_links_items_weight = defaultdict(defaultdict)
        for article_id in self.articles_links:
            for article_link_id in self.articles_links[article_id]:
                if article_id and article_link_id:
                    if article_link_id in articles_weight:
                        articles_links_items_weight[article_id][article_link_id] = articles_weight[article_link_id]
                    if article_id in articles_weight:
                        articles_reverse_links_items_weight[article_link_id][article_id] = articles_weight[article_id]
        for article_id in articles_links_items_weight:
            articles_links_items_weight[article_id] = self._get_top_n_items(articles_links_items_weight[article_id], n)
        for article_id in articles_reverse_links_items_weight:
            articles_reverse_links_items_weight[article_id] = self._get_top_n_items(articles_reverse_links_items_weight[article_id], n)
        return (articles_links_items_weight, articles_reverse_links_items_weight)

    def upload_positions(self, q, method_name, articles_weight):
        logging.info('')
        logging.info('links')

        (articles_links_articles_weight, articles_reverse_links_articles_weight) = self.__count_articles_links_weight(articles_weight, 10)
        articles_links_weights = self._count_weights(articles_links_articles_weight, 0)
        articles_reverse_links_weights = self._count_weights(articles_reverse_links_articles_weight, 0)

        articles_links_positions = self._count_positions(q, articles_links_articles_weight, articles_links_weights, Article.objects, Article.objects)
        articles_reverse_links_positions = self._count_positions(q, articles_reverse_links_articles_weight, articles_reverse_links_weights, Article.objects, Article.objects)

        self._upload_positions(articles_links_positions, method_name + ", type: link")
        self._upload_positions(articles_reverse_links_positions, method_name + ", type: reverse_link")

        return (articles_links_positions, articles_reverse_links_positions)
