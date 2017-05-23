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

    msg_request = 'Please select the desired metric\n' \
                  'e for euclidean\n' \
                  'c for cosine\n' \
                  'q to quit\n'
    metric = ''

    while metric == '':

        metric = input(msg_request)

        if metric == 'q':
            exit()

        elif metric == 'e' or metric == 'c':
            return metric

        else:
            print('Not a valid input\n')
            metric = ''

    return metric


def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

