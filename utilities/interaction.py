from utilities import constants
import readline
import glob
import os


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
