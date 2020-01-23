from collections import defaultdict
from data.models import *
import bisect
import deap.algorithms
import deap.base
import deap.creator
import deap.tools
import logging
import numpy as np
import pickle
import random
import tools.results_presenter
import statistics
import multiprocessing
import tools.graphs

DATA = None
DEBUG = False

def get_articles_positions(individual, question_id, data, debug):
    (articles_id, articles_data, corrected_articles_id, corrected_articles_index) = data[question_id]
    scores = np.sum(articles_data * individual, axis=1)
    # if debug:
    #     logging.debug("question: %d" % question_id)
    #     logging.debug(scores)
    #     logging.debug(scores.max())
    #     logging.debug(scores.min())
    #     logging.debug("corrected articles:")
    positions = []
    for index in corrected_articles_index:
        position = np.count_nonzero(scores >= scores[index])
        positions.append(position)
        # if debug:
        #     logging.debug("  article: %d, position: %d, weight: %.2f" % (article_id, position, article_score))
    # if debug:
    #     logging.debug('-' * 80)
    return positions

def score_p1(individual):
    individual = np.array(individual)
    total_ones = 0
    total_positions = 0
    for question_id in DATA:
        positions = get_articles_positions(individual, question_id, DATA, DEBUG)
        for i in range(1, 10):
            if i in positions:
                total_ones += 1
            else:
                break
        total_positions += len(positions)
    return (total_ones / total_positions, )

def score_mrr(individual):
    individual = np.array(individual)
    total_positions = []
    for question_id in DATA:
        positions = get_articles_positions(individual, question_id, DATA, DEBUG)
        total_positions.extend(positions)
    return (1 / statistics.harmonic_mean(total_positions), )

class EvolutionaryAlgorithm():
    def __init__(self, debug_top_items, workdir, methods_patterns, exclude_methods_patterns, population):
        self.__debug_top_items = debug_top_items
        self.__workdir = workdir
        self.__methods_patterns = methods_patterns
        self.__exclude_methods_patterns = exclude_methods_patterns
        self.__population = population

    def model_name(self):
        return 'ea'

    def __get_methods(self, methods_patterns, exclude_methods_patterns):
        methods_id = set(Method.objects.filter(is_enabled=True).values_list('id', flat=True))
        if methods_patterns:
            for pattern in methods_patterns.split(','):
                methods_id = methods_id & set(Method.objects.filter(name__contains=pattern).values_list('id', flat=True))

        if exclude_methods_patterns:
            exclude_methods_patterns = exclude_methods_patterns.split(',')
            methods_id = list(Method.objects.filter(id__in=methods_id).exclude(method_name__in=exclude_methods_patterns).order_by('name').values_list('id', flat=True))
        else:
            methods_id = list(Method.objects.filter(id__in=methods_id).order_by('name').values_list('id', flat=True))
        logging.info(methods_id)
        methods_id_position = {}
        methods_id_name = {}
        for i in range(0, len(methods_id)):
            method_id = methods_id[i]
            methods_id_position[method_id] = i
            methods_id_name[method_id] = Method.objects.get(id=method_id).name
        return (methods_id, methods_id_position, methods_id_name)

    def __is_smaller_first(self, method_name):
        return 'w2v' in method_name or ('tfi' in method_name and ('cosine' in method_name or 'cityblock' in method_name or 'euclidean' in method_name))

    def __normalise(self, data, method_name):
        max = np.nanmax(data)
        if max != 0.0:
            if self.__is_smaller_first(method_name):
                data /= max
                data *= -1
                data += + 1
            elif 'tfi':
                data /= max

    def __prepare_question(self, question_id, methods_id, methods_id_position, methods_id_name):
        articles_id = list(Rate.objects.filter(question_id=question_id, method_id__in=methods_id).distinct().values_list('article_id', flat=True))
        articles_positions = np.argsort(np.argsort(articles_id))
        articles_id_position = {}
        for i in range(0, len(articles_id)):
            articles_id_position[articles_id[i]] = articles_positions[i]

        articles_data = np.zeros(shape=(len(methods_id), len(articles_id)), dtype=np.float32)
        articles_data.fill(np.nan)
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
                articles_data[method_index][article_index] = weight
            self.__normalise(articles_data[method_index], methods_id_name[method_id])
        articles_data = np.nan_to_num(articles_data)
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
        corrected_articles_index = [articles_id_position[article_id] for article_id in corrected_articles_id]
        return (articles_id, np.transpose(articles_data), corrected_articles_id, corrected_articles_index)

    def __generate_dataset(self, questions, dataset, methods_id, methods_id_position, methods_id_name):
        logging.info("methods: %d" % len(methods_id))
        data = {}

        questions_id = list(map(lambda question: question.id, questions))
        current = 0
        total = len(questions_id)
        step = max(1, int(total / 100))
        for question_id in questions_id:
            data[question_id] = self.__prepare_question(question_id, methods_id, methods_id_position, methods_id_name)
            current += 1
            if current % step == 0:
                logging.debug("progress: %d/%d (%.2f %%)" % (current, total, current / total * 100))

        return data

    def __load_dataset(self, dataset):
        return self.__load_data('ea_' + dataset)

    def __prepare_dataset(self, questions, dataset, methods_id, methods_id_position, methods_id_name):
        try:
            data = self.__load_dataset(dataset)
            logging.info('loading %s succesful' % dataset)
            logging.info('%s size: %d' % (dataset, len(data)))
            return data
        except:
            logging.info('generating new %s' % dataset)
            data = self.__generate_dataset(questions, dataset, methods_id, methods_id_position, methods_id_name)
            self.__save_data(data, 'ea_' + dataset)
            return data

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
        data = self.__load_data('ea_population')
        random.setstate(data["rndstate"])
        return data["population"]

    def __save_population(self, population):
        logging.debug('saving population')
        data = dict(population=population, rndstate=random.getstate())
        self.__save_data(data, 'ea_population')

    def __get_best_score(population, data, train_data):
        global DATA
        DATA = data
        try:
            individual = deap.tools.selBest(population, k=1)[0]
        except AttributeError:
            individual = population
        result = (score_mrr(individual), score_p1(individual))
        DATA = train_data
        return result

    def __solve(self, population, data, method_name, debug_top_items):
        method, created = Method.objects.get_or_create(name=method_name, is_smaller_first=False)
        individual = deap.tools.selBest(population, k=1)[0]
        for question_id in data:
            question = Question.objects.get(id=question_id)
            if not tools.results_presenter.ResultsPresenter.is_already_solved(question, method.name):
                (articles_id, scores, corrected_articles_id, corrected_articles_index) = self.__get_articles_scores(individual, question_id, data)
                tools.results_presenter.ResultsPresenter.present(question, articles_id, scores, method, debug_top_items, False)

    def __test_dataset(population, data, dataset, train_data):
        ((mrr, ), (p1, )) = EvolutionaryAlgorithm.__get_best_score(population, data, train_data)
        logging.info("best individual score_mrr: %.4f, score_p1: %.4f, dataset: %s" % (mrr, p1, dataset))

    def __show_best_individual(population):
        individual = deap.tools.selBest(population, k=1)[0]
        individual = ', '.join(['%.4f' % w for w in individual])
        logging.info('best individual: [%s]' % individual)

    def __make_graph(workdir, population, n, is_better, methods_id, data, train_data):
        individual = deap.tools.selBest(population, k=1)[0]
        ((mrr, ), (p1, )) = EvolutionaryAlgorithm.__get_best_score(population, data, train_data)
        title = 'p: %d, i: %d, MRR: %.4f, p1: %.4f' % (len(population), len(individual), mrr, p1)
        methods = [Method.objects.get(id=id) for id in methods_id]
        labels = list(map(lambda m: m.preety_name(), methods))
        tools.graphs.plot_bar('%s/ea_model_results_%03d_%d.png' % (workdir, n, is_better), individual, labels, title=title)

    def train(self, train_questions, validate_questions, test_questions, epoch):
        if epoch == 0:
            return

        (methods_id, methods_id_position, methods_id_name) = self.__get_methods(self.__methods_patterns, self.__exclude_methods_patterns)
        train_data = self.__prepare_dataset(train_questions, 'train_data', methods_id, methods_id_position, methods_id_name)
        validate_data = self.__prepare_dataset(validate_questions, 'validate_data', methods_id, methods_id_position, methods_id_name)
        test_data = self.__prepare_dataset(test_questions, 'test_data', methods_id, methods_id_position, methods_id_name)
        total_data = {}
        total_data.update(train_data)
        total_data.update(validate_data)
        total_data.update(test_data)

        deap.creator.create("Fitness", deap.base.Fitness, weights=(1.0,))
        deap.creator.create("Individual", list, fitness=deap.creator.Fitness)

        global score_mrr
        global DATA
        DATA = train_data

        toolbox = deap.base.Toolbox()
        toolbox.register("weight", random.uniform, 0.0, 1.0)
        toolbox.register("individual", deap.tools.initRepeat, deap.creator.Individual, toolbox.weight, n=len(methods_id))
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", score_mrr)
        toolbox.register("mate", deap.tools.cxUniform, indpb=0.1)
        toolbox.register("mutate", deap.tools.mutUniformInt, low=0.0, up=1.0, indpb=0.1)
        toolbox.register("select", deap.tools.selTournament, tournsize=50)
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)

        try:
            population = self.__load_population()
            logging.info('load population succesful')
        except Exception as e:
            logging.error(e)
            population = toolbox.population(n=self.__population)
            logging.info('create new population')

        EvolutionaryAlgorithm.__show_best_individual(population)
        EvolutionaryAlgorithm.__make_graph(self.__workdir, population, 0, True, methods_id, validate_data, train_data)
        EvolutionaryAlgorithm.__test_dataset(population, train_data, 'train', train_data)
        EvolutionaryAlgorithm.__test_dataset(population, validate_data, 'validate', train_data)
        EvolutionaryAlgorithm.__test_dataset(population, test_data, 'test', train_data)
        EvolutionaryAlgorithm.__test_dataset(population, total_data, 'total', train_data)
        (best_score_mrr, best_score_p1) = EvolutionaryAlgorithm.__get_best_score(population, validate_data, train_data)

        for e in range(epoch):
            logging.info("generation: %d/%d" % (e+1, epoch))
            offspring = deap.algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.25)
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            current_population = toolbox.select(offspring, k=len(population))
            (current_score_mrr, score_p1) = EvolutionaryAlgorithm.__get_best_score(current_population, validate_data, train_data)
            EvolutionaryAlgorithm.__show_best_individual(current_population)
            EvolutionaryAlgorithm.__test_dataset(current_population, train_data, 'train', train_data)
            EvolutionaryAlgorithm.__test_dataset(current_population, validate_data, 'validate', train_data)
            EvolutionaryAlgorithm.__test_dataset(current_population, test_data, 'test', train_data)
            EvolutionaryAlgorithm.__test_dataset(current_population, total_data, 'total', train_data)
            EvolutionaryAlgorithm.__make_graph(self.__workdir, population, e+1, current_score_mrr > best_score_mrr, methods_id, validate_data, train_data)
            population = current_population
            if current_score_mrr > best_score_mrr:
                best_score_mrr = current_score_mrr
                logging.info('population saved')
                self.__save_population(population)
            else:
                logging.info('population skipped')

        EvolutionaryAlgorithm.__test_dataset(population, train_data, 'train', train_data)
        EvolutionaryAlgorithm.__test_dataset(population, validate_data, 'validate', train_data)
        EvolutionaryAlgorithm.__test_dataset(population, test_data, 'test', train_data)
        EvolutionaryAlgorithm.__test_dataset(population, total_data, 'total', train_data)

    def prepare_for_testing(self):
        self.__population = self.__load_population()
        train_data = self.__load_dataset('train_data')
        validate_data = self.__load_dataset('validate_data')
        test_data = self.__load_dataset('test_data')
        self.__test_dataset(self.__population, train_data, 'train', test_data)
        self.__test_dataset(self.__population, validate_data, 'validate', test_data)
        self.__test_dataset(self.__population, test_data, 'test', test_data)
        (self.__methods_id, self.__methods_id_position, self.__methods_id_name) = self.__get_methods(self.__methods_patterns, self.__exclude_methods_patterns)

    def test(self, question, method_name):
        data = self.__prepare_question(question, self.__methods_id, self.__methods_id_position, self.__methods_id_name)
        self.__solve(self.__population, { question.id : data}, method_name, self.__debug_top_items)
