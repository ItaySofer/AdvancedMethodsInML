# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:35:26 2017

@author: carmonda
"""
import sys
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
from scipy import special

PLOT = True
ALPHA = 0.8
BETA = 0.5

CONVERGENCE_THRESHOLD = 1e-5


def log_phi(value1, value2, coef):
    return coef * value1 * value2


class Vertex(object):
    def __init__(self, name='', y=None, neighs=None, in_msgs=None):
        self._name = name
        self._y = y  # original pixel

        if neighs is None:
            neighs = set()  # set of neighbour nodes

        if in_msgs is None:
            in_msgs = {}  # dictionary mapping neighbours to their messages

        self._neighs = neighs
        self._in_msgs = in_msgs
        self.optional_values = [-1, 1]

    def add_neigh(self, vertex):
        self._neighs.add(vertex)

    def rem_neigh(self, vertex):
        self._neighs.remove(vertex)

    def get_belief(self):
        max_index = np.argmax([log_phi(val, self._y, ALPHA) +
                               np.sum([self.get_in_msgs()[neigh][val] for neigh in self.get_neighbours()])
                               for val in self.optional_values])
        return self.optional_values[max_index]

    def snd_msg(self, neigh):
        """ Combines messages from all other neighbours
            to propagate a message to the neighbouring Vertex 'neigh'.
        """

        my_neighs_excecpt_specific_neigh = [my_neigh for my_neigh in self._neighs if my_neigh._name is not neigh._name]
        for neigh_val in neigh.optional_values:
            neigh.get_in_msgs()[self][neigh_val] = np.max([log_phi(my_val, self._y, ALPHA) +
                                             log_phi(my_val, neigh_val, BETA) +
                                             np.sum([self._in_msgs[my_neigh][my_val] for my_neigh in my_neighs_excecpt_specific_neigh])
                                             for my_val in self.optional_values])

        normalizer = special.logsumexp([float(val) for val in neigh.get_in_msgs()[self].values()])
        for neigh_val in neigh.optional_values:
            neigh.get_in_msgs()[self][neigh_val] -= normalizer

    def get_neighbours(self):
        return self._neighs

    def get_neighbours_sorted(self):
        return sorted(self._neighs, key=lambda v: int(v._name[1:]))

    def get_in_msgs(self):
        return self._in_msgs

    def init_message(self):
        for neigh in self._neighs:
            self._in_msgs[neigh] = {optional_value: 0 for optional_value in self.optional_values}

    def __str__(self):
        ret = "Name: " + self._name
        ret += "\nNeighbours:"
        neigh_list = ""
        for n in self._neighs:
            neigh_list += " " + n._name
        ret += neigh_list
        return ret


class Graph(object):
    def __init__(self, graph_dict=None):
        """ initializes a graph object
            If no dictionary is given, an empty dict will be used
        """
        if graph_dict == None:
            graph_dict = {}
        self._graph_dict = graph_dict

    def vertices(self):
        """ returns the vertices of a graph"""
        return list(self._graph_dict.keys())

    def edges(self):
        """ returns the edges of a graph """
        return self._generate_edges()

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in
            self._graph_dict, a key "vertex" with an empty
            list as a value is added to the dictionary.
            Otherwise nothing has to be done.
        """
        if vertex not in self._graph_dict:
            self._graph_dict[vertex] = []

    def add_edge(self, edge):
        """ assumes that edge is of type set, tuple, or list;
            between two vertices can be multiple edges.
        """
        edge = set(edge)
        (v1, v2) = tuple(edge)
        if v1 in self._graph_dict:
            self._graph_dict[v1].append(v2)
        else:
            self._graph_dict[v1] = [v2]
        # if using Vertex class, update data:
        if (type(v1) == Vertex and type(v2) == Vertex):
            v1.add_neigh(v2)
            v2.add_neigh(v1)

    def generate_edges(self):
        """ A static method generating the edges of the
            graph "graph". Edges are represented as sets
            with one or two vertices
        """
        e = []
        for v in self._graph_dict:
            for neigh in self._graph_dict[v]:
                if {neigh, v} not in e:
                    e.append({v, neigh})
        return e

    def init_messages(self):
        for vertex in self.vertices():
            vertex.init_message()

    def sorted_vertices(self):
        return sorted(self.vertices(), key=lambda v: int(v._name[1:]))

    def get_all_messages(self):
        messages = []
        for vertex in self.sorted_vertices():
            vertex_messages = vertex.get_in_msgs()
            for neigh in vertex.get_neighbours_sorted():
                for val in vertex.optional_values:
                    messages.append(vertex_messages[neigh][val])

        return messages

    def __str__(self):
        res = "V: "
        for k in self._graph_dict:
            res += str(k) + " "
        res += "\nE: "
        for edge in self._generate_edges():
            res += str(edge) + " "
        return res


def build_grid_graph(n, m, img_mat):
    """ Builds an nxm grid graph, with vertex values corresponding to pixel intensities.
    n: num of rows
    m: num of columns
    img_mat = np.ndarray of shape (n,m) of pixel intensities
    
    returns the Graph object corresponding to the grid
    """
    V = []
    g = Graph()
    # add vertices:
    for i in range(n * m):
        row, col = (i // m, i % m)
        v = Vertex(name="v" + str(i), y=img_mat[row][col])
        g.add_vertex(v)
        if ((i % m) != 0):  # has left edge
            g.add_edge((v, V[i - 1]))
        if (i >= m):  # has up edge
            g.add_edge((v, V[i - m]))
        V += [v]
    return g


def grid2mat(grid, n, m):
    """ convertes grid graph to a np.ndarray
    n: num of rows
    m: num of columns
    
    returns: np.ndarray of shape (n,m)
    """
    mat = np.zeros((n, m))
    l = grid.vertices()  # list of vertices
    for v in l:
        i = int(v._name[1:])
        row, col = (i // m, i % m)
        mat[row][col] = v.get_belief()
    return mat


def run_lbp(g):
    g.init_messages()
    vertices_sort = g.sorted_vertices()

    prev_messages = None
    curr_messages = g.get_all_messages()
    while prev_messages is None or not np.allclose(prev_messages, curr_messages, atol=CONVERGENCE_THRESHOLD):
        for vertex in vertices_sort:
            for neigh in vertex.get_neighbours():
                vertex.snd_msg(neigh)

        prev_messages = curr_messages
        curr_messages = g.get_all_messages()

def main():
    # begin:
    if len(sys.argv) < 3:
        print('Please specify input and output file names.')
        exit(0)
    # load image:
    in_file_name = sys.argv[1]
    image = misc.imread(in_file_name + '.png')
    n, m = image.shape

    # binarize the image.
    image = image.astype(np.float32)
    image[image < 128] = -1.
    image[image > 127] = 1.
    if PLOT:
        plt.imshow(image)
        plt.show()

    # build grid:
    g = build_grid_graph(n, m, image)

    # process grid:
    run_lbp(g)

    # convert grid to image: 
    infered_img = grid2mat(g, n, m)
    if PLOT:
        plt.imshow(infered_img)
        plt.show()

    # save result to output file
    out_file_name = sys.argv[2]
    misc.toimage(infered_img).save(out_file_name + '.png')


if __name__ == "__main__":
    main()
