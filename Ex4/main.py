import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import util
import vis
from scipy.sparse.csgraph import minimum_spanning_tree

np.set_printoptions(precision=4)
pd.set_option('precision', 2)


class Marginals(object):
    def __init__(self, count_dict, M):
        self._count_dict = count_dict
        self._M = M

    def get_single_marginal(self, label, value):
        freq = float(self._count_dict[label]) / self._M

        return freq if value == 1 else 1 - freq

    def get_pair_marginal(self, label_a, label_b, value_a, value_b):
        pair_freq = float(self._count_dict[order_pair((label_a, label_b))]) / self._M
        freq_a = float(self._count_dict[label_a]) / self._M
        freq_b = float(self._count_dict[label_b]) / self._M

        if value_a == 1 and value_b == 1:
            return pair_freq
        if value_a == 1 and value_b == 0:
            return freq_a - pair_freq
        if value_a == 0 and value_b == 1:
            return freq_b - pair_freq

        return 1 - freq_a - freq_b + pair_freq


def add_single_count(labels, counts):
    for label in labels:
        counts[label] = counts[label] + 1
        
    return counts


def create_all_pairs(labels):
    pairs = list()
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            pairs.append((labels[i], labels[j]))

    return pairs


def order_pair(pair):
    if pair[0] <= pair[1]:
        return pair
    else:
        return (pair[1], pair[0])


def add_pair_count(labels, counts):
    for pair in create_all_pairs(labels):
        ordered_pair = order_pair(pair)
        counts[ordered_pair] = counts[ordered_pair] + 1
        
    return counts


def compute_marginals(samples):
    counts = defaultdict(int)
    for sample in samples:
        add_single_count(sample, counts)
        add_pair_count(sample, counts)
        
    return Marginals(counts, len(samples))


def calculate_mutual_information(label_a, label_b, marginals):
    values = [(0, 0), (0, 1), (1, 0), (1, 1)]
    mi = 0
    for (xi, xj) in values:
        pair_pD = marginals.get_pair_marginal(label_a=label_a, label_b=label_b, value_a=xi, value_b=xj)
        if pair_pD == 0:
            continue
        pD_a = marginals.get_single_marginal(label_a, xi)
        pD_b = marginals.get_single_marginal(label_b, xj)

        mi += pair_pD * np.log(pair_pD / (pD_a * pD_b))

    return mi


def create_mi_matrix(marginals, labels_in_data, label_to_index):
    matrix = np.zeros((len(label_to_index), len(label_to_index)), dtype=np.float)

    for i in range(len(labels_in_data)):
        for j in range(i+1, len(labels_in_data)):
            first_label = labels_in_data[i]
            second_label = labels_in_data[j]
            mi = calculate_mutual_information(first_label, second_label, marginals)

            matrix[label_to_index[first_label]][label_to_index[second_label]] = -mi

    return matrix


def compute_tree_dictionary(mst_matrix, index_to_label, labels, labels_in_data):
    tree = defaultdict(list)

    # compute from mst matrix
    rows, columns = mst_matrix.nonzero()
    for (i, j) in zip(rows, columns):
        first_label = index_to_label[i]
        second_label = index_to_label[j]

        tree[first_label].append(second_label)
        tree[second_label].append(first_label)

    # add labels that are not in the data
    for label in labels:
        if label not in labels_in_data:
            tree[label] = []

    return tree


def run_chew_liu(image_to_labels, classes_display_name):
    marginals = compute_marginals(image_to_labels.values())
    labels_in_data = list(set(itertools.chain.from_iterable(image_to_labels.values())))
    labels = classes_display_name.keys()
    label_to_index = {label: i for i, label in enumerate(labels)}
    index_to_label = {i: label for i, label in enumerate(labels)}
    mi_matrix = create_mi_matrix(marginals, labels_in_data, label_to_index)

    x = csr_matrix(mi_matrix)
    mst_matrix = minimum_spanning_tree(x)
    return compute_tree_dictionary(mst_matrix, index_to_label, labels, labels_in_data)


def main():
    vocabolary_threshold = 400
    oid_data = 'data/annotations-machine.csv'
    classes_fn = 'data/class-descriptions.csv'

    # Mapping between class lable and class name
    classes_display_name = util.load_display_names(classes_fn)
    image_to_labels = util.image_to_labels(pd.read_csv(oid_data))

    # Dictionary with mapping between each Node and its childern nodes.
    # use for each node the class lable
    chow_liu_tree = run_chew_liu(image_to_labels, classes_display_name)

    vis.plot_network(chow_liu_tree, classes_display_name)


if __name__ == '__main__':
    main()
