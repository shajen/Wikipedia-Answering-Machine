from data.models import *
from django import db

import argparse
import logging
import multiprocessing
import os
import queue
import shlex
import subprocess
import sys
sys.path.append(os.path.dirname(__file__))

import calculators.deep_averaging_neural_weight_calculator
import calculators.evolutionary_algorithm
import calculators.neural_weight_calculator
import calculators.tf_idf_weight_calculator
import calculators.weight_comparator
import calculators.word2vec_weight_calculator
import tools.data_loader
import tools.logger
import tools.results_presenter

def resolve_questions_tf_idf(args, questions_queue, method, data_loader):
    neighbors = list(map(lambda x: int(x), args.neighbors.split(',')))
    minimal_word_idf_weights = list(map(lambda x: float(x), args.minimal_word_idf_weights.split(',')))
    power_factors = list(map(lambda x: int(x), args.power_factors.split(',')))

    tf_idf_calculator = calculators.tf_idf_weight_calculator.TfIdfWeightCalculator(args.debug_top_items, args.ngram, data_loader)

    while True:
        try:
            question = questions_queue.get(timeout=1)
            prepared = False
            for neighbor in neighbors:
                for minimal_word_idf_weight in minimal_word_idf_weights:
                    comparators = []
                    method_name = '%s, mwiw: %.2f' % (method, minimal_word_idf_weight)
                    if args.tfidf_models:
                        for power_factor in power_factors:
                            comparators.append(calculators.weight_comparator.TfIdfWeightComparator(method_name, neighbor, power_factor))
                    if args.vector_model_cosine:
                        comparators.append(calculators.weight_comparator.CosineWeightComparator(method_name, neighbor))
                    if args.vector_model_cityblock:
                        comparators.append(calculators.weight_comparator.CityblockWeightComparator(method_name, neighbor))
                    if args.vector_model_euclidean:
                        comparators.append(calculators.weight_comparator.EuclideanWeightComparator(method_name, neighbor))

                    comparators = list(filter(lambda comparator: not tools.results_presenter.ResultsPresenter.is_already_solved(question, comparator.method()), comparators))
                    if comparators:
                        if not prepared:
                            tf_idf_calculator.prepare(question, args.title)
                            prepared = True
                        tf_idf_calculator.calculate(question, neighbor, minimal_word_idf_weight, comparators)
        except queue.Empty:
            break

def resolve_questions_word2vec(args, questions_queue, method_name, data_loader):
    word2vec_calculator = calculators.word2vec_weight_calculator.Word2VecWeightCalculator(args.debug_top_items, data_loader)
    while True:
        try:
            question = questions_queue.get(timeout=1)
            if not tools.results_presenter.ResultsPresenter.is_already_solved(question, method_name):
                word2vec_calculator.calculate(question, method_name, args.title, args.topn)
        except queue.Empty:
            break

def start_callback_threads(args, questions, method_name, callback, callback_arguments):
    db.connections.close_all()
    logging.info('method: %s' % method_name)
    questions_queue = multiprocessing.Queue()
    for question in questions:
        questions_queue.put(question)

    if args.threads == 1:
        callback(*((args, questions_queue, method_name) + callback_arguments))
    else:
        try:
            threads = []
            for i in range(args.threads):
                thread = multiprocessing.Process(target=callback, args=(args, questions_queue, method_name) + callback_arguments)
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            logging.info('stopping threads')
            for thread in threads:
                thread.terminate()

def split_questions(questions, args):
    values = list(map(lambda n: int(n), args.dataset_proportion.split(':')))
    train_dataset_count = round(len(questions) * values[0] / sum(values))
    validate_dataset_count = round(len(questions) * values[1] / sum(values))
    train_dataset = questions[:train_dataset_count]
    validate_dataset = questions[train_dataset_count:train_dataset_count+validate_dataset_count]
    test_dataset = questions[train_dataset_count+validate_dataset_count:]
    logging.info('train dateset size    : %d' % len(train_dataset))
    logging.info('validate dateset size : %d' % len(validate_dataset))
    logging.info('test dateset size     : %d' % len(test_dataset))
    logging.info('total questions size  : %d' % len(questions))
    logging.info('dataset sum size      : %d' % (len(train_dataset) + len(validate_dataset) + len(test_dataset)))
    return (train_dataset, validate_dataset, test_dataset)

def start_learning_model(args, questions, model, method_name):
    logging.info('start learning model: %s' % model.model_name())
    logging.info('method: %s' % method_name)
    (train_questions, validate_questions, test_questions) = split_questions(questions, args)
    logging.info('start training model: %s' % model.model_name())
    model.train(train_questions, validate_questions, test_questions, args.epoch)
    if not args.disable_testing:
        logging.info('prepare for testing model: %s' % model.model_name())
        model.prepare_for_testing()
        logging.info('testing model: %s' % model.model_name())
        for question in questions:
            if not tools.results_presenter.ResultsPresenter.is_already_solved(question, method_name):
                model.test(question, method_name)

def get_workdir(workdir, name):
    workdir = '%s/%s/' % (workdir, 'ea')
    if not os.path.isdir(workdir):
        os.mkdir(workdir)
    return workdir

def start(args, questions, method_name):
    learning_model_count = (args.learning_model_questions_words_count, args.learning_model_articles_title_words_count, args.learning_model_articles_content_words_count)
    classic_model_count = (args.classic_model_questions_words_count, args.classic_model_articles_title_words_count, args.classic_model_articles_content_words_count)
    learning_count = 'q: %02d, at: %02d, ac: %04d' % learning_model_count
    data_loader = tools.data_loader.DataLoader(learning_model_count, classic_model_count, args.word2vec_file, args.word2vec_size, args.word2vec_random, args.extend_stop_words)

    if any([args.tfidf_models, args.vector_model_cosine, args.vector_model_cityblock, args.vector_model_euclidean]):
        start_callback_threads(args, questions, '%s, type: tfi, title: %d, ngram: %d' % (method_name, args.title, args.ngram), resolve_questions_tf_idf, (data_loader,))
    if args.word2vec_model:
        data_loader.load_word2vec_model()
        start_callback_threads(args, questions, '%s, type: w2v, topn: %03d, title: %d, %s' % (method_name, args.topn, args.title, learning_count), resolve_questions_word2vec, (data_loader,))
    if args.convolution_neural_network:
        model = calculators.neural_weight_calculator.NeuralWeightCalculator(data_loader, args.debug_top_items, get_workdir(args.cache_directory, 'nn'), args.neural_model_good_bad_ratio, args.neural_model_method)
        start_learning_model(args, questions, model, '%s, type: cnn, %s' % (method_name, learning_count))
    if args.deep_averaging_network:
        model = calculators.deep_averaging_neural_weight_calculator.DeepAveragingNeuralWeightCalculator(data_loader, args.debug_top_items, get_workdir(args.cache_directory, 'nn'), args.neural_model_good_bad_ratio, args.neural_model_method)
        start_learning_model(args, questions, model, '%s, type: dan, %s' % (method_name, learning_count))
    if args.evolutionary_algorithm:
        model = calculators.evolutionary_algorithm.EvolutionaryAlgorithm(args.debug_top_items, get_workdir(args.cache_directory, 'ea'), args.evolutionary_algorithm_methods_patterns, args.evolutionary_algorithm_exclude_methods_patterns, args.evolutionary_algorithm_population)
        start_learning_model(args, questions, model, '%s, type: ean, p: %04d' % (method_name, args.evolutionary_algorithm_population))
    logging.info('finish')

def get_method_name(args):
    dirPath = os.path.dirname(os.path.realpath(__file__))
    commit_hash = subprocess.check_output(['git', '-C', dirPath, 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    commit_datetime = subprocess.check_output(['git', '-C', dirPath, 'log', '-1', '--format=%at']).decode('ascii').strip()
    method_name = 'git: %s_%s' % (commit_datetime, commit_hash)
    if args.method:
        method_name = 'name: %s, %s' % (args.method, method_name)
    return method_name

def run(*args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    logging.getLogger("gensim").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("smart_open.smart_open_lib").setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.INFO)

    try:
        args = shlex.split(args[0])
    except IndexError:
        args = []
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threads", help="threads", type=int, default=1, choices=range(1, 129), metavar="int")
    parser.add_argument("-q", "--questions", help="solve first n questions", type=int, default=10**9, metavar="n")
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('-m', '--method', help="method name to make unique in database", type=str, default='', metavar="method")
    parser.add_argument("-dti", "--debug_top_items", help="print top n items in debug", type=int, default=3, metavar="int")
    parser.add_argument("-tn", "--topn", help="use n nearest words in word2vec model", type=int, default=10, metavar="n")
    parser.add_argument("-cd", "--cache_directory", help="directory to save and read cache data", type=str)
    parser.add_argument("-esw", "--extend_stop_words", help="use base forms of stop words as stop words", action='store_true')
    parser.add_argument("-w2vf", "--word2vec_file", help="path to word2vec model", type=str, default='', metavar="file")
    parser.add_argument("-w2vs", "--word2vec_size", help="size of word2vec vector", type=int, default=100, metavar="n")
    parser.add_argument("-w2vr", "--word2vec_random", help="randomize word vector if not found in dataset", action='store_true')

    parser.add_argument("-T", "--title", help="calculate based on articles title not content", action='store_true')
    parser.add_argument('-n', '--ngram', help="use ngram mode", type=int, default=1, choices=[1, 2], metavar="ngram")
    parser.add_argument("-N", "--neighbors", help="count tf-idf in every n neighboring words tuple in articles", type=str, default='0', metavar="0,10,20")
    parser.add_argument("-mwiw", "--minimal_word_idf_weights", help="use only words with idf weight above", type=str, default='0.0', metavar="0.0,1.6,3.2")
    parser.add_argument("-pf", "--power_factors", help="use to sum words weight in count article weight", type=str, default='3', metavar="1,2,3,4")
    parser.add_argument("-cm_qwc", "--classic_model_questions_words_count", help="use first n words from questions", type=int, default=100)
    parser.add_argument("-cm_atwc", "--classic_model_articles_title_words_count", help="use first n words from articles title", type=int, default=100)
    parser.add_argument("-cm_acwc", "--classic_model_articles_content_words_count", help="use first n words from articles", type=int, default=1000)

    parser.add_argument("-dp", "--dataset_proportion", help="proportion of train:validate:test dataset", type=str, default='60:20:20')
    parser.add_argument("-e", "--epoch", help="train n epoch", type=int, default=10)
    parser.add_argument("-dt", "--disable_testing", help="train models without testing", action='store_true')
    parser.add_argument("-lm_qwc", "--learning_model_questions_words_count", help="use first n words from questions", type=int, default=20)
    parser.add_argument("-lm_atwc", "--learning_model_articles_title_words_count", help="use first n words from articles title", type=int, default=20)
    parser.add_argument("-lm_acwc", "--learning_model_articles_content_words_count", help="use first n words from articles", type=int, default=100)

    parser.add_argument("-tm", "--tfidf_models", help="use td-idf models", action='store_true')
    parser.add_argument("-vm_cos", "--vector_model_cosine", help="use cosine vector models", action='store_true')
    parser.add_argument("-vm_city", "--vector_model_cityblock", help="use cityblock vector models", action='store_true')
    parser.add_argument("-vm_e", "--vector_model_euclidean", help="use euclidean vector models", action='store_true')
    parser.add_argument("-w2vm", "--word2vec_model", help="use word2vec model", action='store_true')
    parser.add_argument("-cnn", "--convolution_neural_network", help="use onvolution neural network", action='store_true')
    parser.add_argument("-dan", "--deep_averaging_network", help="use deep averaging network", action='store_true')
    parser.add_argument("-nm_gbr", "--neural_model_good_bad_ratio", help="ratio between good and bad articles", type=int, default=1)
    parser.add_argument("-nm_m", "--neural_model_method", help="use method id to choose article id", type=int, default=0)
    parser.add_argument("-ea", "--evolutionary_algorithm", help="enable evolutionary algorithm model", action='store_true')
    parser.add_argument("-ea_p", "--evolutionary_algorithm_population", help="population size", type=int, default=100)
    parser.add_argument("-ea_mp", "--evolutionary_algorithm_methods_patterns", help="methods patterns used in model", type=str, default='', metavar="method1,method2")
    parser.add_argument("-ea_emp", "--evolutionary_algorithm_exclude_methods_patterns", help="methods exclude patterns used in model", type=str, default='', metavar="method1,method2")
    args = parser.parse_args(args)
    tools.logger.configLogger(args.verbose)
    for arg in vars(args):
        logging.info("%s: %s" % (arg.ljust(50), getattr(args, arg)))

    questions = list(Question.objects.order_by('id').all())[:args.questions]
    start(args, questions, get_method_name(args))
