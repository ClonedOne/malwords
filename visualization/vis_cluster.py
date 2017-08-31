from matplotlib import pyplot as plt
from collections import defaultdict
from utilities import interaction
from utilities import constants
import plotly.graph_objs as go
import plotly.offline as oly
from plotly import tools
import seaborn as sns
import numpy as np
import json
import os


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


def plot_cluster_features(config, clustering):
    """
    Plot the histograms of the features of the clusters.
    For each cluster, order the features and plot the histograms, then move down.

    :param config: application configuration dictionary
    :param clustering: dictionary mapping uuids to cluster ids
    :return:
    """

    dir_store = config['dir_store']
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    word_list = sorted(list(words.keys()))

    print('Plotting clustering features')

    reverse_clustering = defaultdict(list)
    for uuid, cluster in clustering.items():
        reverse_clustering[cluster].append(uuid)

    i = 1
    n_clust = len(reverse_clustering)
    base = np.arange(len(word_list))

    axis_dict = dict(
        autorange=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        autotick=True,
        ticks='',
        showticklabels=False
    )
    fig = tools.make_subplots(rows=n_clust, cols=1)

    for cluster in sorted(reverse_clustering):
        cluster_features = np.zeros(len(word_list))

        uuids = reverse_clustering[cluster]
        for uuid in uuids:
            tfidfs = json.load(open(os.path.join(dir_store, uuid), 'r'))

            for j in range(len(word_list)):
                cluster_features[j] += tfidfs.get(word_list[j], 0)

        trace = go.Scatter(x=base, y=cluster_features)
        fig.append_trace(trace, i, 1)
        fig['layout']['xaxis{}'.format(i)].update(axis_dict)
        fig['layout']['yaxis{}'.format(i)].update(axis_dict)

        i += 1

    oly.plot(fig, filename='stacked-subplots')
