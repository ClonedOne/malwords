from plotly.graph_objs import *
import plotly.offline as ply
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

    aggregate = {ind: ([], []) for ind in sbl}

    i = 0
    for point in data:
        ind = base_labels[i]
        aggregate[ind][0].append(point[0])
        aggregate[ind][1].append(point[1])
        i += 1

    traces = []
    for ind, group in aggregate.items():
        trace = Scatter(
            x=group[0],
            y=group[1],
            name=ind,
            mode='markers',
            marker=Marker(
                size=10,
                opacity=0.8
            )
        )
        traces.append(trace)

    ply.iplot(traces, filename='2dData')


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

    trace1 = Scatter3d(
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
    layout = Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = Figure(data=data, layout=layout)
    ply.iplot(fig, filename='3dData')
