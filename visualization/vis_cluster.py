from matplotlib import pyplot as plt
from collections import defaultdict
from utilities import constants
import plotly.graph_objs as go
import plotly.offline as oly
from utilities import utils
from plotly import tools
import seaborn as sns
import numpy as np
import json
import os


def plot_clustering(data_matrix, uuid_pos, y_pred):
    """
    Plot the result of clustering over a dimensionality reduced space.

    :param data_matrix: 2d data matrix file
    :param uuid_pos: rows of the data matrix corresponding to clustered uuids
    :param y_pred: labels resulting from clustering
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


def plot_cluster_features(config, clustering, names=None):
    """
    Plot the histograms of the features of the clusters.
    For each cluster, order the features and plot the histograms, then move down.

    :param config: application configuration dictionary
    :param clustering: dictionary mapping uuids to cluster ids
    :param names: family labels
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

        if names:
            name = names[i - 1]
        else:
            name = str(i)

        trace = go.Scatter(x=base, y=cluster_features, name=name)
        fig.append_trace(trace, i, 1)
        fig['layout']['xaxis{}'.format(i)].update(axis_dict)
        fig['layout']['yaxis{}'.format(i)].update(axis_dict)

        i += 1

    oly.plot(fig, filename='stacked-subplots')


def plot_av_features(uuids, config):
    """
    Uses the cluster feature plotting method to show the features of the clusters provided by AV labeling.

    :param uuids: list of uuids
    :param config: configuration dictionary
    :return:
    """

    labels = utils.get_base_labels_uuids(uuids)
    pseudo_clustering = dict(zip(uuids, labels))
    families = utils.get_index_labels()
    families = sorted(set([families[label] for label in labels]))

    plot_cluster_features(config, pseudo_clustering, families)
