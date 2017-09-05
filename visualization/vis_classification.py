from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from utilities import utils
import seaborn as sns
import numpy as np


def plot_classification(data_matrix, uuid_pos, y_pred, y_true):
    """
    Plots the result of classification against the base truth, over a dimensionality reduced space.
    The base truth will be shown by the center color while the classification will be reflected by the edge color.

    :param data_matrix:
    :param uuid_pos:
    :param y_pred:
    :param y_true:
    :return:
    """

    print('Plotting classification results')

    data = np.loadtxt(data_matrix)
    data = np.array([data[pos] for pos in uuid_pos])

    color_palette = sns.color_palette('bright', max(y_true) + 1)

    c_true = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in y_true]
    c_pred = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in y_pred]

    plt.scatter(*data.T, s=100, linewidth=2, c=c_true, edgecolors=c_pred, alpha=0.8)
    plt.show()


def plot_confusion_matrix(base_labels, predicted_labels):
    """
    Generates and plots the confusion matrix for a given classification.

    :param base_labels:
    :param predicted_labels:
    :return:
    """

    index_labels = utils.get_index_labels()
    families = [index_labels[label] for label in sorted(set(base_labels))]

    cm = confusion_matrix(base_labels, predicted_labels).astype(float)
    for vec in cm:
        vec /= np.sum(vec)

    sns.heatmap(cm, annot=True, xticklabels=families, yticklabels=families)

    plt.show()
