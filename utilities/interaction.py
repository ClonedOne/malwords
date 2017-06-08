from utilities import constants
import readline
import glob
import os


def ask_clusters(num_clusters_max):
    """
    Ask user for the desired number of clusters.

    :param num_clusters_max: maximum number of clusters
    :return: selected number of clusters
    """

    num_clusters = 0

    while num_clusters == 0:
        num_clusters = input('Please select the number of clusters\n')
        try:
            num_clusters = int(num_clusters)
            if num_clusters > num_clusters_max or num_clusters < 2:
                raise Exception
        except:
            num_clusters = 0
            print('Please insert a valid number of clusters\n')

    return num_clusters


def ask_number(request):
    """
    Ask user for the number of requested items.

    :return: 
    """

    msg_request = 'Please select the desired number of {} (q to quit)\n'.format(request)
    number = 0

    while number == 0:

        number = input(msg_request)

        if number == 'q':
            exit()

        try:
            number = int(number)

        except:
            print('Not a valid input\n')
            number = 0

    return number


def ask_file(request):
    """
    Ask user for the file path of requested items.

    :return: 
    """

    msg_request = 'Please select the desired {} (q to quit)\n'.format(request)
    file_path = ""

    while file_path == "":

        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind("tab: complete")
        readline.set_completer(complete)

        file_path = input(msg_request)

        if file_path == 'q':
            exit()

        if not os.path.isfile(file_path):
            print('Not a valid input\n')
            file_path = ""

    return file_path


def ask_metric():
    """
    Ask user for a specific metric to use.

    :return: 
    """

    possible_metrics = ['e', 'c', 'j']
    metric = ''

    while metric == '':

        metric = input(constants.msg_metric)

        if metric == 'q':
            exit()

        elif metric in possible_metrics:
            return metric

        else:
            print('Not a valid input\n')
            metric = ''

    return metric


def complete(text, state):
    return (glob.glob(text + '*') + [None])[state]
