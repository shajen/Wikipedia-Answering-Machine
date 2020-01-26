from data.models import *
import matplotlib.pyplot as plt

def plot_bar(filename, data, labels, **kwargs):
    fontsize = kwargs.get('fontsize', 5)
    title = kwargs.get('title', '')
    reverse = kwargs.get('reverse', True)
    plt.clf()

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

    plt.subplots_adjust(left=0.35, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(filename, dpi=300)

def autolabel(rects, ax, **kwargs):
    fontsize = kwargs.get('fontsize', 5)
    format = kwargs.get('format', '%.2f')
    for rect in rects:
        height = rect.get_height()
        ax.annotate(format % height, xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=fontsize)

def plot_multibar(filename, data, labels, group_labels, **kwargs):
    fontsize = kwargs.get('fontsize', 5)
    title = kwargs.get('title', '')
    x_label = kwargs.get('x_label', '')
    y_label = kwargs.get('y_label', '')
    y_scale = kwargs.get('y_scale', None)
    width = 1 / (len(data) + 1)
    plt.clf()

    x = np.arange(len(group_labels))
    fig, ax = plt.subplots()
    rects = []
    for i in range(len(data)):
        rects.append(ax.bar(x + i * width + width / 2 - width * len(data) / 2, data[i], width, label=labels[i]))

    if title:
        ax.set_title(title, fontsize=fontsize)
    if x_label:
        ax.set_xlabel(x_label, fontsize=fontsize)
    if y_label:
        ax.set_ylabel(y_label, fontsize=fontsize)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend(fontsize=fontsize)
    for rect in rects:
        autolabel(rect, ax, **kwargs)
    fig.tight_layout()
    if y_scale:
        plt.setp(ax, ylim=y_scale)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.savefig(filename, dpi=300)
