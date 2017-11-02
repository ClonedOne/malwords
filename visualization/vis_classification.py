from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from utilities import utils
import seaborn as sns
import numpy as np



def plot_confusion_matrix(base_labels, predicted_labels):
    """
    Generates and plots the confusion matrix for a given classification.

    :param base_labels:
    :param predicted_labels:
    :return:
    """
