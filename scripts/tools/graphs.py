from data.models import *
import matplotlib.pyplot as plt
import os

def plot_bar(filename, data, labels, **kwargs):
    fontsize = kwargs.get('fontsize', 5)
    title = kwargs.get('title', '')
    reverse = kwargs.get('reverse', True)

    if reverse:
        data.reverse()
        labels.reverse()

    pos = list(range(len(data)))

    plt.figure(figsize=(5, 8))
    plt.grid(axis='x', linestyle='--')
    plt.barh(pos, data)
    plt.yticks(pos, labels, fontsize=fontsize)
    if title:
        plt.title(title)

    try:
        os.remove(filename)
    except:
        pass
    plt.subplots_adjust(left=0.35, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(filename, dpi=300)
    plt.clf()
