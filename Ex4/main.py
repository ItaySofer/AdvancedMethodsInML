import numpy as np
import pandas as pd
from sol import utils
import vis
from scipy.sparse.csgraph import minimum_spanning_tree

np.set_printoptions(precision=4)
pd.set_option('precision', 2)


def main():
    vocabolary_threshold = 400
    oid_data = 'data/annotations-machine.csv'
    classes_fn = 'data/class-descriptions.csv'

    # Mapping between class lable and class name
    classes_display_name = utils.load_display_names(classes_fn)

    #####################
    # ADD YOUR CODE HERE#
    ####################

    # Dictionary with mapping between each Node and its childern nodes.
    # use for each node the class lable
    chow_liu_tree = dict()

    vis.plot_network(chow_liu_tree, classes_display_name)


if __name__ == '__main__':
    main()
