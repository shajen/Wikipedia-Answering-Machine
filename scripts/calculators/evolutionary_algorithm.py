from collections import defaultdict
from data.models import *
import bisect
import deap.algorithms
import deap.base
import deap.creator
import deap.tools
import logging
import numpy as np
import random
import pickle

class EvolutionaryAlgorithm():
    def __init__(self, workdir):
        self.__workdir = workdir

    def __get_methods(self, methods_patterns):
        methods_id = set(Method.objects.order_by('id').values_list('id', flat=True))
        if methods_patterns:
            for pattern in methods_patterns.split(','):
                methods_id = methods_id & set(Method.objects.filter(name__contains=pattern).values_list('id', flat=True))

        methods_id = list(methods_id)
        methods_id = sorted(methods_id)
        methods_id_position = {}
        methods_id_name = {}
        for i in range(0, len(methods_id)):
            method_id = methods_id[i]
            methods_id_position[method_id] = i
            methods_id_name[method_id] = Method.objects.get(id=method_id).name
        return (methods_id, methods_id_position, methods_id_name)

    def __normalise(self, data, method_name):
        max = data.max()
        if max != 0.0:
            if 'w2v' in method_name:
                data /= max
                data *= -1
                data += + 1
            elif 'tfi':
                data /= max

    def __prepare_question(self, question_id, methods_id, methods_id_position, methods_id_name):
        articles_id = Rate.objects.filter(question_id=question_id, method_id__in=methods_id).distinct().values_list('article_id', flat=True)
        articles_positions = np.argsort(np.argsort(articles_id))
        articles_id_position = {}
        for i in range(0, len(articles_id)):
            articles_id_position[articles_id[i]] = articles_positions[i]

        articles_data = np.zeros(shape=(len(methods_id), len(articles_id)), dtype=np.float32)
        answers = Answer.objects.filter(question_id=question_id)
        for method_id in methods_id:
            corrected_articles_position = {}
            for answer in answers:
                try:
                    corrected_articles_position[answer.article_id] = Solution.objects.get(answer=answer, method_id=method_id).position
                except:
                    # logging.error('method: %d, question: %d, article: %d' % (method_id, question_id, answer.article_id))
                    pass
            for (article_id, weight) in Rate.objects.filter(question_id=question_id, method_id=method_id).values_list('article_id', 'weight'):
                if article_id in corrected_articles_position and corrected_articles_position[article_id] > 100:
                    # logging.error('method: %d, question: %d, article: %d' % (method_id, question_id, article_id))
                    continue
                article_index = articles_id_position[article_id]
                method_index = methods_id_position[method_id]
                if weight == 100000.0: #TF-IDF model fix
                    articles_data[method_index][article_index] = 0.0
                else:
                    articles_data[method_index][article_index] = weight
            self.__normalise(articles_data[method_index], methods_id_name[method_id])
        # self.__train_data[question_id] = (articles_id, np.transpose(articles_data))
        # logging.debug("question: %d" % question_id)
        # logging.debug("articles: %d" % len(articles_id))
        # logging.debug("articles shape: %s" % str(articles_data.shape))
        # logging.debug("articles_data min: %.2f, max: %.2f" % (articles_data.min(), articles_data.max()))
        # if question_id in [11977]:
        #     for method_id in methods_id:
        #         method_index = methods_id_position[method_id]
        #         method_name = methods_id_name[method_id]
        #         logging.debug("  min: %.2f, max: %.2f, %s" % (articles_data[method_index].min(), articles_data[method_index].max(), method_name))
        corrected_articles_id = list(Answer.objects.filter(question_id=question_id).values_list('article_id', flat=True))
        return (articles_id, np.transpose(articles_data), corrected_articles_id)

    def __prepare_dataset(self, questions_id, dataset, methods_id, methods_id_position, methods_id_name):
        try:
            data = self.__load_data(dataset)
            logging.info('loading %s succesful' % dataset)
            logging.info('%s size: %d' % (dataset, len(data)))
            return data
        except:
            logging.info('generating new %s' % dataset)

        logging.info("methods: %d" % len(methods_id))
        data = {}

        current = 0
        total = len(questions_id)
        step = max(1, int(total / 100))
        for question_id in questions_id:
            data[question_id] = self.__prepare_question(question_id, methods_id, methods_id_position, methods_id_name)
            current += 1
            if current % step == 0:
                logging.debug("progress: %d/%d (%.2f %%)" % (current, total, current / total * 100))

        self.__save_data(data, dataset)
        return data

    def __get_articles_scores(self, individual, question_id, data):
        (articles_id, articles_data, corrected_articles_id) = data[question_id]
        scores = np.sum(articles_data * individual, axis=1)
        return (articles_id, scores, corrected_articles_id)

    def __get_articles_positions(self, individual, question_id, data, debug):
        # if debug:
        #     logging.debug("question: %d" % question_id)
        #     logging.debug("corrected articles:")
        (articles_id, scores, corrected_articles_id) = self.__get_articles_scores(individual, question_id, data)
        positions = []
        for article_id in corrected_articles_id:
            position = bisect.bisect_left(articles_id, article_id)
            article_score = scores[position]
            position = 1
            for score in scores:
                if score > article_score:
                    position += 1
            positions.append(position)
            # if debug:
            #     logging.debug("  article: %d, position: %d" % (article_id, position))
        return positions

    def __score(self, individual, data, debug):
        individual = np.array(individual)
        total_ones = 0
        for question_id in data:
            positions = self.__get_articles_positions(individual, question_id, data, debug)
            if 1 in positions:
                total_ones += 1
        return total_ones / len(data)

    def __save_data(self, data, name):
        filename = '%s/%s.pkl' % (self.__workdir, name)
        logging.debug('saving %s' % filename)
        with open(filename, "wb") as cp_file:
            pickle.dump(data, cp_file)

    def __load_data(self, name):
        filename = '%s/%s.pkl' % (self.__workdir, name)
        logging.debug('loading %s' % filename)
        with open(filename, "rb") as file:
            return pickle.load(file)

    def __load_population(self):
        logging.debug('loading population')
        data = self.__load_data('population')
        random.setstate(data["rndstate"])
        return data["population"]

    def __save_population(self, population):
        logging.debug('saving population')
        data = dict(population=population, rndstate=random.getstate())
        self.__save_data(data, 'population')

    def __get_best_score(self, population, data):
        individual = deap.tools.selBest(population, k=1)[0]
        return self.__score(individual, data, True)

    def run(self, train_questions_id, test_questions_id, method_name, debug_top_items, methods_patterns, population, generations):
        (methods_id, methods_id_position, methods_id_name) = self.__get_methods(methods_patterns)
        train_data = self.__prepare_dataset(train_questions_id, 'train_data', methods_id, methods_id_position, methods_id_name)
        test_data = self.__prepare_dataset(test_questions_id, 'test_data', methods_id, methods_id_position, methods_id_name)
        deap.creator.create("Fitness", deap.base.Fitness, weights=(1.0,))
        deap.creator.create("Individual", list, fitness=deap.creator.Fitness)

        toolbox = deap.base.Toolbox()
        toolbox.register("weight", random.uniform, 0.0, 1.0)
        toolbox.register("individual", deap.tools.initRepeat, deap.creator.Individual, toolbox.weight, n=len(methods_id))
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", lambda individual: (self.__score(individual, train_data, False),) )
        toolbox.register("mate", deap.tools.cxTwoPoint)
        toolbox.register("mutate", deap.tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", deap.tools.selTournament, tournsize=3)

        try:
            population = self.__load_population()
            logging.info('load population succesful')
        except Exception as e:
            logging.error(e)
            population = toolbox.population(n=population)
            logging.info('create new population')

        best_score = self.__get_best_score(population, train_data)
        logging.info("best population score: %.2f, dataset: train" % best_score)
        logging.info("best population score: %.2f, dataset: test" % self.__get_best_score(population, test_data))

        for gen in range(generations):
            logging.debug("generation: %d" % gen)
            offspring = deap.algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            population = toolbox.select(offspring, k=len(population))
            score = self.__get_best_score(population, train_data)
            logging.info("best population score: %.2f, dataset: train" % score)
            logging.info("best population score: %.2f, dataset: test" % self.__get_best_score(population, test_data))
            if score > best_score:
                self.__save_population(population)
