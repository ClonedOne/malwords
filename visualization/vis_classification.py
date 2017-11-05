from sklearn.metrics import confusion_matrix
import plotly.graph_objs as go
import plotly.offline as ply
import numpy as np


def plot_net_costs(costs):
    """
    Plots the costs obtained during the network training phase
    :param costs: list of costs
    :return:
    """
    trace = go.Scatter(
        x=np.arange(len(costs)),
        y=costs
    )
    ply.iplot([trace], filename='costs')


def plot_confusion_matrix(y_test, y_test_fam, y_predicted):
    """
    Generates and plots the confusion matrix for a given classification.

    :param y_test: test labels
    :param y_test_fam: test labels with mnemonic
    :param y_predicted: predicted labels
    :return:
    """

    classes = sorted(set(y_test))
    n_classes = len(classes)

    classes_dict = dict(zip(classes, range(n_classes)))

    class_fam = {}
    for i in range(len(y_test_fam)):
        class_fam[classes_dict[y_test[i]]] = y_test_fam[i]

    cm = confusion_matrix(y_test, y_predicted).astype(float)
    for vec in cm:
        vec /= np.sum(vec)

    families = [class_fam[i] for i in sorted(class_fam.keys())]

    trace = go.Heatmap(z=cm, x=families, y=families)
    ply.iplot([trace], filename='conf_matrix')
