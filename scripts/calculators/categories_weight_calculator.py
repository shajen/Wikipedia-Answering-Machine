from collections import Counter
from collections import defaultdict
from data.models import *
import calculators.weight_calculator
import logging

class CategoriesWeightCalculator(calculators.weight_calculator.WeightCalculator):
    def __init__(self, debug_top_items):
        super().__init__(debug_top_items)
        logging.info('start reading categories')
        self.categories_articles = defaultdict(set)
        for data in Category.objects.values('id', 'article'):
            self.categories_articles[data['id']].add(data['article'])
        logging.info('finish reading')

    def __count_articles_categories_weight(self, categories_items_weight, n, power_factor):
        for category_id in categories_items_weight:
            categories_items_weight[category_id] = self._get_top_n_items(categories_items_weight[category_id], n)
        categories_weight = self._count_weights(categories_items_weight, power_factor)

        if self.debug_top_items > 0:
            categories_ranking = Counter(categories_weight).most_common()
            logging.info('top %d categories:' % self.debug_top_items)
            for (id, weight) in categories_ranking[:self.debug_top_items]:
                self._print_item(id, categories_weight, categories_items_weight, Category.objects, Article.objects)

        articles_categories_weight = defaultdict(defaultdict)
        for category_id in self.categories_articles:
            for article_id in self.categories_articles[category_id]:
                articles_categories_weight[article_id][category_id] = categories_weight[category_id]
        for article_id in articles_categories_weight:
            articles_categories_weight[article_id] = self._get_top_n_items(articles_categories_weight[article_id], n)
        return articles_categories_weight

    def __count_categories_weight(self, articles_weight, n):
        categories_items_weight = defaultdict(defaultdict)
        for category_id in self.categories_articles:
            for article_id in self.categories_articles[category_id]:
                if article_id in articles_weight:
                    categories_items_weight[category_id][article_id] = articles_weight[article_id]
            categories_items_weight[category_id] = self._get_top_n_items(categories_items_weight[category_id], n)
        return categories_items_weight

    def upload_positions(self, q, method_name, articles_weight):
        logging.info('')
        logging.info('categories')

        categories_articles_weight = self.__count_categories_weight(articles_weight, 10)
        articles_categories_weight = self.__count_articles_categories_weight(categories_articles_weight, 10, 0)
        articles_weights = self._count_weights(articles_categories_weight, 0)
        articles_categories_positions = self._count_positions(q, articles_categories_weight, articles_weights, Article.objects, Category.objects)

        self._upload_positions(articles_categories_positions, method_name + ", type: categories")
        return articles_categories_positions
