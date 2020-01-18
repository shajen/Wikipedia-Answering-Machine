from data.models import *

import logging
import sys
from collections import defaultdict
from termcolor import colored, cprint
from functools import reduce

class ReportManager():
    def __init__ (self, no_color):
        self.no_color = no_color

    def split_questions(questions, dataset_proportion):
        values = list(map(lambda n: int(n), dataset_proportion.split(':')))
        train_dataset_count = round(len(questions) * values[0] / sum(values))
        validate_dataset_count = round(len(questions) * values[1] / sum(values))
        train_dataset = questions[:train_dataset_count]
        validate_dataset = questions[train_dataset_count:train_dataset_count+validate_dataset_count]
        test_dataset = questions[train_dataset_count+validate_dataset_count:]
        return (train_dataset, validate_dataset, test_dataset)

    def methodColor(self, name):
        if self.no_color:
            return name
        return colored(name, 'yellow', attrs={"bold"})

    def splitColor(self, splitString):
        if self.no_color:
            return splitString
        return colored(splitString, 'red', attrs={"bold"})

    def process(self, args):
        logging.info("process with top answers (%s)" % ', '.join(str(x) for x in args['tops']))
        questions_id = Question.objects.values_list('id', flat=True)
        if args['questionCount'] > 0:
            questions_id = list(questions_id)[:args['questionCount']]
        if args['dataset_proportion']:
            (train_dataset, validate_dataset, test_dataset) = ReportManager.split_questions(questions_id, args['dataset_proportion'])
            self.printErrorRate(args, "train dataset: %d" % len(train_dataset), train_dataset)
            self.printErrorRate(args, "validate dataset: %d" % len(validate_dataset), validate_dataset)
            self.printErrorRate(args, "test dataset: %d" % len(test_dataset), test_dataset)
        else:
            self.printErrorRate(args, 'all', questions_id)
        logging.info("finish")

    def printErrorRate(self, args, name, questions_id):
        if args['all']:
            methods = Method.objects.filter(name__contains=args['methodPatterns']).order_by('name')
        else:
            methods = Method.objects.filter(name__contains=args['methodPatterns'], is_enabled=True).order_by('name')
        LEN = 105
        sys.stdout.write(name.ljust(LEN + 9) + '     #')
        for t in args['tops']:
            sys.stdout.write('  %6d' % t)
        sys.stdout.write('MRP'.rjust(8))
        sys.stdout.write('\n')
        for method in methods:
            sort_sign = colored('<', 'green', attrs={"bold"}) if method.is_smaller_first else colored('>', 'red', attrs={"bold"})
            (answersCount, scores) = method.scores(questions_id, args['tops'])
            if len(method.name) <= LEN:
                sys.stdout.write("#%5d %s %s %5d" % (method.id, sort_sign, self.methodColor(method.name.ljust(LEN)), answersCount))
            else:
                sys.stdout.write("#%5d %s %s\n" % (method.id, sort_sign, self.methodColor(method.name)))
                sys.stdout.write('%s %5d' % (' ' * LEN, answersCount))
            for score in scores:
                sys.stdout.write("  %.4f" % score)
            sys.stdout.write('\n')
