from data.models import *
import matplotlib.pyplot as plt
import matplotlib
import os

def plot(global_pattern, patterns, fontsize, i):
    data = []
    labels = []
    for pattern in patterns:
        for method in Method.objects.filter(is_enabled=True).filter(name__contains=pattern).filter(name__contains=global_pattern).order_by('name'):
            (count, scores) = method.scores()
            data.append(scores[i])
            labels.append(method.preety_name())

    pos = list(range(len(data)))
    plt.figure(figsize=(5, 8))
    plt.grid(axis='x', linestyle='--')
    plt.barh(pos, data)
    plt.yticks(pos, labels, fontsize=fontsize)

    filename = '/home/shajen/mgr/praca/images/model_summary_%d.png' % i
    try:
        os.remove(filename)
    except:
        pass
    plt.subplots_adjust(left=0.35, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(filename, dpi = 300)
    plt.clf()

def run(*args):
    global_pattern = 'final'
    patterns = ['pf', 'w2v', 'cosine', 'euclidean', 'cityblock']
    patterns.reverse()
    fontsize = 5
    plot(global_pattern, patterns, fontsize, 0)
    plot(global_pattern, patterns, fontsize, 1)
    plot(global_pattern, patterns, fontsize, 2)
    plot(global_pattern, patterns, fontsize, 3)
