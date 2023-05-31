#!/usr/bin/env python3
"""
#####################################
# Part of the PATH FINDER  Module   #
# Author J.Yellen                   #
#####################################
"""
import numpy as np

class Graph():

    def __init__(self):
        self._adj = {}
        self._node = {}

    def __construct_nodes(self, edge:list) -> None:
        for item in edge:
            if item[0] not in self._node:
                self._node[item[0]] = {}
                self._adj[item[0]] = {}

    def __construct_adj(self, edge:list) -> None:
        self.__construct_nodes(edge)
        for item in edge:
            source, child , weight = item
            if child not in self._adj[source]:
                self._adj[source][child] = {}
            index = len(self._adj[source][child])
            self._adj[source][child] = {index: {'weight': weight}}

    def add_weighted_edges(self, edges:list, clear:bool=False) -> None:
        if clear:
            self._adj= {}
            self._node = {}
        self.__construct_adj(edges)

    def edges(self, srce:int=None)-> list:
        if isinstance(srce, list):
            srce = srce[0]
        if srce is None:
            return [(k, i) for k, subdict in self._adj.items() for i in subdict]
        if srce in self._adj:
            return [(srce, i) for i in self._adj[srce]]
        return []

class BinaryAcceptance(Graph):

    def __init__ (self, matrix:np.ndarray[bool, float],
                  weights:list|None = None,
                  threshold:float|None = None) -> None:
        super().__init__()
        self.source = 0
        self.bin_acc = self.set_binary_acceptance(matrix, threshold)
        self.weights = self.set_weights(weights, self.dim)
        self.set_weighted_graph()

    @property
    def dim(self) -> int:
        return self.bin_acc.shape[0]

    # setter method
    @staticmethod
    def set_binary_acceptance(matrix:np.ndarray, threshold:float|None=None)-> np.ndarray:
        if matrix.ndim != 2:
            raise ValueError('Binary acceptance is not a 2d array')

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Binary acceptance is not square')

        if matrix.dtype == bool:
            return matrix
        elif matrix.dtype == 'int' and threshold is None:
            print(f'{10*"#"} Warning! {10*"#"} \n\n Binary acceptance \
is array of integers, converting format to True/False')
            return np.array(abs(matrix), dtype=bool)
        elif threshold is not None:
            return abs(matrix) < threshold
        else:
            raise ValueError('Binary acceptance is not Boolean type!, \
\n Convert or provide threshold')

    # setter method
    @staticmethod
    def set_weights(weights:list|None, size:int)-> np.ndarray:
        if weights is None:
            weights = [1] * size
        else:
            weights = list(weights)
        if len(weights) == size:
            weights += [0.0]
        return np.array(weights)

    @staticmethod
    def set_dummy_target(bin_acc:np.ndarray, source:int)-> np.ndarray:
        """
        Set up the binary acceptence matrix setting the
        source diagonal element to True.
        """
        dim = np.array(bin_acc.shape)
        bin_acc_plus = np.ones(dim+1, dtype=bool)
        bin_acc_plus[:-1:, :-1:] = bin_acc
        bin_acc_plus[source, source] = True
        # These lines make things fast!
        sub = ~bin_acc_plus[source, :]
        bin_acc_plus[:, sub] = False
        bin_acc_plus[sub, :] = False
        return bin_acc_plus

    @staticmethod
    def strip_subdict(dct:dict, target:str)->list:
        return [p[target] for _, p in dct.items()]

    def get_full_triu(self) -> np.ndarray:
        """
        upper tri of binary acceptance with dummy target
        """
        full_graph_mat = self.set_dummy_target(self.bin_acc, self.source)
        return np.triu(full_graph_mat, 1)

    def set_weighted_graph(self) -> None:
        """
        add weights to the node edges
        """
        edges= [(*ij, self.weights[ij[1]]) for ij in np.argwhere(self.get_full_triu())]
        self.add_weighted_edges(edges, clear=True)

    def get_weight(self, path:list) -> float:
        """ Get the sum of the weights for a given path of indices"""
        if path:
            return np.sum(self.weights[path])
        return 0.0

    def reset_source(self, source:int=0)-> None:
        """
        reset the source node
        """
        if source >= self.dim:
            print('Source out of range Defaulting to zero')
            source = 0
        self.source = source
        self.set_weighted_graph()
