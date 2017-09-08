from dimensionality_reduction import dr_action
from classification import cla_action
from visualization import vis_action
from preprocessing import pp_action
from clustering import clu_action
from keywords import kw_action
import json


def main():
    config = json.load(open('config.json', 'r'))

    uuids, base_labels = pp_action.pre_process(config)

    x_train, x_test, y_train, y_test = pp_action.show_data(uuids, base_labels)

    dr_action.dimensionality_reduction(uuids, x_train, x_test, config)

    clu_action.cluster(uuids, base_labels, config)

    cla_action.classify(x_train, x_test, y_train, y_test, config)

    vis_action.visualize(config, uuids, base_labels)

    kw_action.keywords_extraction(config)


if __name__ == '__main__':
    main()
