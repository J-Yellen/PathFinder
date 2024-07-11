#!/usr/bin/env python3
"""
#####################################
# Part of the PATH FINDER  Module   #
# Author J.Yellen                   #
#####################################
"""
import numpy as np
from typing import Optional, Union, Annotated, TypeVar, Literal

ScalarType_co = TypeVar("ScalarType_co", bound=np.generic, contravariant=True)
Array1D_Float = Annotated[np.ndarray[ScalarType_co, np.float_], Literal['N']]
Array1D_int = Annotated[np.ndarray[ScalarType_co, np.int_], Literal['N']]
Array1D_bool = Annotated[np.ndarray[ScalarType_co, bool], Literal['N']]
Array2D_Float = Annotated[np.ndarray[ScalarType_co, np.float_], Literal['N', 'N']]
Array2D_Bool = Annotated[np.ndarray[ScalarType_co, bool], Literal['N', 'N']]


class Graph():

    def __init__(self):
        self._adj = {}
        self._node = {}

    def __construct_nodes(self, edge: list) -> None:
        for item in edge:
            if item[0] not in self._node:
                self._node[int(item[0])] = {}
                self._adj[int(item[0])] = {}

    def __construct_adj(self, edge: list) -> None:
        self.__construct_nodes(edge)
        for item in edge:
            source, child, weight = item
            if child not in self._adj[source]:
                self._adj[int(source)][int(child)] = {}
            index = len(self._adj[source][child])
            self._adj[source][child] = {index: {'weight': weight}}

    def add_weighted_edges(self, edges: list, clear: bool = False) -> None:
        if clear:
            self._adj = {}
            self._node = {}
        self.__construct_adj(edges)

    def edges(self, srce: int = None) -> list:
        if isinstance(srce, list):
            srce = srce[0]
        if srce is None:
            return [(k, i) for k, subdict in self._adj.items() for i in subdict]
        if srce in self._adj:
            return [(srce, i) for i in self._adj[srce]]
        return []


class BinaryAcceptance(Graph):

    def __init__(self, matrix: Union[Array2D_Bool, Array2D_Float], weights: Optional[list] = None,
                 labels: Optional[list] = None, threshold: Optional[float] = None,
                 allow_negative_weights: bool = False) -> None:
        super().__init__()

        self.source = 0
        self.bin_acc = self.set_binary_acceptance(matrix, threshold)
        self.weights = self.set_weights(weights, self.dim)
        self.labels = labels
        if not allow_negative_weights and min(self.weights) < 0.0:
            raise Exception('WARNING! Negative weights provided. Rescale or set allow_negative_weights to True')
        self.set_weighted_graph()

    @property
    def dim(self) -> int:
        return self.bin_acc.shape[0]

    @property
    def get_source_row(self) -> Array1D_bool:
        return self.bin_acc[self.source, :]

    @property
    def get_source_row_index(self) -> Array1D_int:
        return np.where(self.get_source_row)[0][self.source + 1::]

    # setter method
    @staticmethod
    def set_binary_acceptance(matrix: Union[Array2D_Bool, Array2D_Float],
                              threshold: Optional[float] = None) -> Array2D_Bool:
        if matrix.ndim != 2:
            raise ValueError('Binary acceptance is not a 2d array')

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Binary acceptance is not square')

        if matrix.dtype == bool:
            return matrix
        elif matrix.dtype == 'int' and threshold is None:
            print(f'{10*"#"} Warning! {10*"#"}\nBinary acceptance \
is array of integers, converting format to True/False')
            return np.array(abs(matrix), dtype=bool)
        elif threshold is not None:
            return abs(matrix) < threshold
        else:
            raise ValueError('Binary acceptance is not Boolean type!, Convert or provide threshold')

    # setter method
    @staticmethod
    def set_weights(weights: Optional[list], size: int) -> Array1D_Float:
        if weights is None:
            weights = [1] * size
        else:
            weights = list(weights)
        if len(weights) == size:
            weights += [0.0]
        return np.array(weights)

    @staticmethod
    def set_dummy_target(bin_acc: Array2D_Bool, source: int) -> Array2D_Bool:
        """
        Set up the binary acceptence matrix setting the
        source diagonal element to True.
        """
        dim = np.array(bin_acc.shape)
        bin_acc_plus = np.ones(dim + 1, dtype=bool)
        bin_acc_plus[:-1:, :-1:] = bin_acc
        bin_acc_plus[source, source] = True
        # These lines make things fast!
        sub = ~bin_acc_plus[source, :]
        bin_acc_plus[:, sub] = False
        bin_acc_plus[sub, :] = False
        return bin_acc_plus

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
        edges = [(*ij, self.weights[ij[1]]) for ij in np.argwhere(self.get_full_triu())]
        self.add_weighted_edges(edges, clear=True)

    def get_weight(self, path: list) -> float:
        """ Get the sum of the weights for a given path of indices"""
        if path:
            return np.sum(self.weights[path])
        return 0.0

    def reset_source(self, source: int = 0) -> None:
        """
        reset the source node
        """
        if source >= self.dim:
            print('Source out of range. Defaulting to zero')
            source = 0
        self.source = source
        self.set_weighted_graph()

    def sort_bam_by_weight(self) -> Array1D_int:
        index_map = np.argsort(self.weights[:-1:])[::-1].astype(int)
        self.weights[:-1:] = self.weights[index_map]
        self.bin_acc = self.bin_acc[index_map, :][:, index_map]
        if self.labels is not None:
            self.labels = [self.labels[i] for i in index_map]
        return index_map
