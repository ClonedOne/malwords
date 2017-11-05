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

    sbl = sorted(set(base_labels))

    aggregate = {ind: ([], [], []) for ind in sbl}

    i = 0
    for point in data:
        ind = base_labels[i]
        aggregate[ind][0].append(point[0])
        aggregate[ind][1].append(point[1])
        aggregate[ind][2].append(point[2])
        i += 1

    traces = []
    for ind, group in aggregate.items():
        trace = Scatter3d(
            x=group[0],
            y=group[1],
            z=group[2],
            name=ind,
            mode='markers',
            marker=Marker(
                opacity=0.8
            )
        )
        traces.append(trace)

    ply.iplot(traces, filename='3dData')


def take_malfie(tfidfs, words):
    """
    Generate a picture using the sample features as pixel data.

    :param tfidfs: weighted tf-idf values for a sample
    :param words: dictionary of words and the related id
    :return:
    """

    mat_size = len(words)
    mat = np.zeros(mat_size)

    for word, pos in words.items():
        mat[int(pos)] = float(tfidfs.get(word, 0))

    mat = mat.reshape(-1, 500)

    print(mat.shape)

    trace = Heatmap(z=mat)
    ply.iplot([trace], filename='malfie')
