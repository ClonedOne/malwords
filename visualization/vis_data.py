import matplotlib.pyplot as plt
import plotly.graph_objs as go
from utilities import utils
import plotly.plotly as py
import seaborn as sns
import numpy as np


def plot_data(data_matrix, base_labels):
    """
    Plot dimensionality reduced representations of data.

    :param data_matrix: 2d/3d data matrix file
    :param base_labels: reference classification by AV
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
    
    :param base_labels: reference classification by AV
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
    
    :param base_labels: reference classification by AV
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
