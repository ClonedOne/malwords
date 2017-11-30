from utilities import constants
import readline
import glob
import os


def ask_number(msg, min_valid=None, max_valid=None):
    """
    Ask user for the number of requested items.

    :param msg: message to be displayed
    :param min_valid: optional minimum value
    :param max_valid: optional maximum value
    :return:
    """

    msg_request = 'Please select the desired number of {} (q to quit)\n'.format(msg)
    number = None

    while number is None:

        number = input(msg_request)

        if number == 'q':
            exit()

        try:
            number = int(number)

            if (min_valid and number < min_valid) or (max_valid and number > max_valid):
                print('Not a valid input\n')
                number = None

        except ValueError:
            print('Not a valid input\n')
            number = None

    return number


def ask_file(msg):
    """
    Ask user for the file path of requested items.

    :param msg: message to be displayed
    :return: 
    """

    msg_request = 'Please select the desired {} (q to quit)\n'.format(msg)
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

    possible_metrics = ['e', 'c', 'c1', 'j', 'j1']
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


def ask_yes_no(msg):
    """
    Ask user for a yes/no answer.

    :param msg: message to be displayed
    :return:
    """

    chosen = False
    while not chosen:
        choice = input(msg)

        if choice.lower() == 'y':
            return True

        elif choice.lower() == 'n':
            return False

        else:
            print('Not a valid input\n')


def ask_action(msg, possibilities):
    """
    Asks the user to chose one of a set of possible actions.

    :param msg: message to be displayed
    :param possibilities: set of possible inputs
    :return:
    """

    chosen = None
    while not chosen:
        chosen = input(msg)

        if chosen in possibilities or chosen == 's':
            return chosen

        elif chosen == 'q':
            exit()

        else:
            print('Not a valid input\n')
            chosen = None


def complete(text, state):
    return (glob.glob(text + '*') + [None])[state]
