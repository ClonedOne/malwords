import matplotlib.pyplot as plt
import plotly.graph_objs as go
from utilities import utils
import plotly.plotly as py
import seaborn as sns
import numpy as np
import os


def plot_data(data_matrix, base_labels):
    """
    Plot dimensionality reduced representations of data.

    :return: 
    """

    data = np.loadtxt(data_matrix)

    print('Number of labels:', len(set(base_labels)))

    if data.shape[1] == 2:
        plot2d(base_labels, data)
    elif data.shape[1] == 3:
        plot3d(base_labels, data)
    else:
        print('Wrong data shape')


def plot2d(base_labels, data):
    """
    Plot data in 2 dimensions.
    
    :param base_labels: reference clustering by AV
    :param data: 2d data matrix
    :return: 
    """

    sbl = sorted(set(base_labels))

    color_palette = sns.color_palette('bright', len(base_labels))
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in sbl]

    print('Number of colors:', len(cluster_colors))

    index_labels = utils.get_index_labels()
    labels = [index_labels[i] if i >= 0 else 'noise' for i in sbl]

    xs = {}
    ys = {}
    for i in range(len(base_labels)):
        xs[base_labels[i]] = xs.get(base_labels[i], []) + [data.T[0][i], ]
        ys[base_labels[i]] = ys.get(base_labels[i], []) + [data.T[1][i], ]

    for i in range(len(sbl)):
        plt.scatter(xs[sbl[i]], ys[sbl[i]], c=cluster_colors[i], alpha=0.5, label=labels[i])

    plt.legend(loc=3)
    plt.show()


def plot3d(base_labels, data):
    """
    Plot data in 3 dimensions.
    
    :param base_labels: reference clustering by AV
    :param data: 3d data matrix
    :return: 
    """

    dt = data.T
    print(dt.shape)
    x = dt[0]
    y = dt[1]
    z = dt[2]
    print(type(x), len(x), type(y), len(y), type(z), len(z))

    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=base_labels,  # set color to an array/list of desired values
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )

    data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='3d-scatter-colorscale')


def plot_hdbs_against_2d(hdbs, num_clusters):
    """
    Plot the clustering result of HDBSCAN against a 2d projection of data.

    :return:
    """

    matrix_file2d = ""

    # Forces the user to choose the desired number of clusters
    while matrix_file2d == "":
        matrix_file2d = input('Please select the 2d data file or q to quit\n')

        if matrix_file2d == 'q':
            exit()

        try:
            if not os.path.isfile(matrix_file2d):
                raise Exception
        except:
            matrix_file2d = ""
            print('Please select the 2d data file or q to quit\n')

    data_red = np.loadtxt(matrix_file2d)

    color_palette = sns.color_palette('bright', num_clusters)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in hdbs.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, hdbs.probabilities_)]
    plt.scatter(*data_red.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.5)
    plt.show()


def plot_classification(data_matrix, classification, base_labels):
    """
    Plots the result of classification against the base truth, over a dimensionality reduced space.

    :param data_matrix:
    :param classification:
    :param base_labels:
    :return:
    """

    print('Plotting classification results')

    data = np.loadtxt(data_matrix)
    color_palette = sns.color_palette('deep', max(base_labels) + 1)

    colors_base = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in base_labels]
    colors_classified = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in classification]

    plt.scatter(*data.T, s=60, linewidth=2, c=colors_base, edgecolors=colors_classified, alpha=0.8)
    plt.show()
