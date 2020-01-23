from data.models import *
import os
import sys

sys.path.append(os.path.dirname(__file__))

import tools.graphs

def run(*args):
    methods = Method.objects.filter(is_enabled=True).order_by('name')
    labels = list(map(lambda m: m.preety_name(), methods))
    for i in range(0, 4):
        filename = '/home/shajen/mgr/praca/images/model_summary_%d.png' % i
        data = list(map(lambda m: m.scores()[1][i], methods))
        tools.graphs.plot_bar(filename, data, labels)
