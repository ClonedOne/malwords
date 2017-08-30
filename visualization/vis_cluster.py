from matplotlib import pyplot as plt
from utilities import interaction
from utilities import constants
import seaborn as sns
import numpy as np


def plot_hdbs_against_2d(hdbs, num_clusters):
    """
    Plot the clustering result of HDBSCAN against a 2d projection of data.

    :return:
    """

    matrix_file2d = interaction.ask_file(constants.msg_vis_base)

    data_red = np.loadtxt(matrix_file2d)

    color_palette = sns.color_palette('bright', num_clusters)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in hdbs.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, hdbs.probabilities_)]
    plt.scatter(*data_red.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.5)
    plt.show()


def plot_clustering(data_matrix, uuid_pos, y_pred):
    """
    Plot the result of clustering over a dimensionality reduced space.

    :param data_matrix:
    :param uuid_pos:
    :param y_pred:
    :return:
    """

    print('Plotting clustering results')

    data = np.loadtxt(data_matrix)
    data = np.array([data[pos] for pos in uuid_pos])

    num_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    color_palette = sns.color_palette('bright', num_clusters)

    c_clu = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in y_pred]

    plt.scatter(*data.T, s=60, c=c_clu, alpha=0.8)
    plt.show()
