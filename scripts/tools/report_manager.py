from data.models import *

import logging
import sys
from collections import defaultdict
from termcolor import colored, cprint
from functools import reduce

class ReportManager():
    def __init__ (self, no_color):
        self.no_color = no_color

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
        self.printErrorRate(args)
        self.printQuestions(args)
        logging.info("finish")

    def calculateMethodsQuestionsPositions(self, args):
        methods_questions_positions = defaultdict(lambda: defaultdict(set))
        question_id = 0
        for solution in Solution.objects.values('position', 'method_id'):#$all():
            position = solution['position']
            method_id = solution['method_id']
            # question_id = solution.answer.question.id
            question_id += 1 #just for acceleration
            if not args['showNotFound'] and position == 10**9:
                continue
            methods_questions_positions[method_id][question_id].add(position)
        return methods_questions_positions

    def printErrorRate(self, args):
        methods_questions_positions = self.calculateMethodsQuestionsPositions(args)
        LEN = 105
        sys.stdout.write(' ' * (LEN + 7) + '     #')
        for t in args['tops']:
            sys.stdout.write('  %6d' % t)
        sys.stdout.write('\n')
        for method in Method.objects.filter(name__contains=args['methodPatterns']).order_by('name').all():
            answersCount = method.solution_set.count()
            if answersCount == 0:
                continue
            positions = methods_questions_positions[method.id].values()
            positions = [list(x) for x in positions]
            positions = reduce(lambda x, y: x + y, positions, [])
            if all(len(list(filter(lambda x: x <= t, positions)))/answersCount <= args['hideScoreTreshold'] for t in args['tops']):
                continue
            if len(method.name) <= LEN:
                sys.stdout.write("#%5d %s %5d" % (method.id, self.methodColor(method.name.ljust(LEN)), answersCount))
            else:
                sys.stdout.write("#%5d %s\n" % (method.id, self.methodColor(method.name)))
                sys.stdout.write('%s %5d' % (' ' * LEN, answersCount))

            for t in args['tops']:
                corrected = len(list(filter(lambda x: x <= t, positions)))
                weight = corrected/answersCount
                if (weight < args['hideScoreTreshold']):
                    sys.stdout.write("        ")
                else:
                    sys.stdout.write("  %.4f" % weight)
            sys.stdout.write('\n')

    def printQuestions(self, args):
        if args['questionCount'] <= 0:
            return
        print('')
        articles = defaultdict(set)
        for method in Method.objects.order_by('name').all():
            print("method %s" % self.methodColor(method.name))
            data = defaultdict(list)
            questions_positions = defaultdict(set)
            for solution in method.solution_set.all():
                questions_positions[solution.answer.question.id].add(solution.position)
            for question_id in questions_positions:
                positions = list(questions_positions[question_id])
                data[max(positions)].append((question_id, positions))
            printedCount = 0
            for pos, value in sorted(data.items(), reverse=not args['questionBetter']):
                if (args['questionBetter'] and pos < args['questionAnswerPosition']) or (not args['questionBetter'] and pos > args['questionAnswerPosition']):
                    continue
                for (question_id, positions) in value:
                    if printedCount >= args['questionCount']:
                        break
                    question = Question.objects.get(id=question_id)
                    print("#QT %s" % question.name)
                    for answer in question.answer_set.all():
                        print("#AT %s" % answer.article.title)
                    print("#QI %s" % question_id)
                    for answer in question.answer_set.all():
                        print("#AI %s" % answer.article.id)
                    print("#AP %s" % ', '.join(str(p) for p in positions))
                    print(self.splitColor("-" * 80))
                    printedCount += 1
                if printedCount >= args['questionCount']:
                    break
            print("")
