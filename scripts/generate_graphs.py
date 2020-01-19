from data.models import *
import matplotlib.pyplot as plt
import matplotlib
import os

def plot(fontsize, i):
    methods = Method.objects.filter(is_enabled=True).order_by('-name')
    filename = '/home/shajen/mgr/praca/images/model_summary_%d.png' % i
    data = list(map(lambda m: m.scores()[1][i], methods))
    plot_data(filename, data, fontsize)

def plot_data(filename, data, fontsize):
    methods = Method.objects.filter(is_enabled=True).order_by('-name')
    labels = list(map(lambda m: m.preety_name(), methods))
    pos = list(range(len(data)))

    plt.figure(figsize=(5, 8))
    plt.grid(axis='x', linestyle='--')
    plt.barh(pos, data)
    plt.yticks(pos, labels, fontsize=fontsize)

    try:
        os.remove(filename)
    except:
        pass
    plt.subplots_adjust(left=0.35, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(filename, dpi = 300)
    plt.clf()

def run(*args):
    fontsize = 5
    plot(fontsize, 0)
    plot(fontsize, 1)
    plot(fontsize, 2)
    plot(fontsize, 3)
    data = list(range(22))
    plot_data('/home/shajen/mgr/praca/images/model_ea.png', data, fontsize)
