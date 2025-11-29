#!/usr/bin/env python3
"""
#####################################
# Part of the PATH FINDER  Module   #
# Author J.Yellen                   #
#####################################
"""
import numpy as np
from typing import Optional, Union, Annotated, TypeVar, Literal, List, Dict, Tuple

ScalarType_co = TypeVar("ScalarType_co", bound=np.generic, contravariant=True)
Array1D_Float = Annotated[np.ndarray[tuple[int, ...], np.dtype[np.float64]], Literal['N']]
Array1D_int = Annotated[np.ndarray[tuple[int, ...], np.dtype[np.int64]], Literal['N']]
Array1D_bool = Annotated[np.ndarray[tuple[int, ...], np.dtype[np.bool_]], Literal['N']]
Array2D_Float = Annotated[np.ndarray[tuple[int, int], np.dtype[np.float64]], Literal['N', 'N']]
Array2D_Bool = Annotated[np.ndarray[tuple[int, int], np.dtype[np.bool_]], Literal['N', 'N']]


class Graph():
    """
    Weighted directed graph implementation using adjacency list representation.

    Used internally by BinaryAcceptance to represent pairwise acceptance relationships.
    """

    def __init__(self) -> None:
        """Initialise empty graph with no nodes or edges."""
        self._adj: Dict[int, Dict[int, Dict[int, Dict[str, float]]]] = {}
        self._node: Dict[int, Dict] = {}

    def _construct_nodes(self, edge: List[Tuple[int, int, float]]) -> None:
        """
        Create node entries in graph from edge list.

        Args:
            edge: List of tuples (source, target, weight) defining edges.

        Returns:
            None
        """
        for item in edge:
            if item[0] not in self._node:
                self._node[int(item[0])] = {}
                self._adj[int(item[0])] = {}

    def _construct_adj(self, edge: List[Tuple[int, int, float]]) -> None:
        """
        Build adjacency list from edge list.

        Args:
            edge: List of tuples (source, target, weight) defining edges.

        Returns:
            None
        """
        self._construct_nodes(edge)
        for item in edge:
            source, child, weight = item
            if child not in self._adj[source]:
                self._adj[int(source)][int(child)] = {}
            index = len(self._adj[source][child])
            self._adj[source][child] = {index: {'weight': weight}}

    def add_weighted_edges(self, edges: List[Tuple[int, int, float]], clear: bool = False) -> None:
        """
        Add weighted edges to the graph.

        Args:
            edges: List of tuples (source, target, weight) defining edges to add.
            clear: If True, clear existing graph before adding edges. Default False.

        Returns:
            None
        """
        if clear:
            self._adj = {}
            self._node = {}
        self._construct_adj(edges)

    def edges(self, srce: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Get edges from specified source node or all edges if source is None.

        Args:
            srce: Source node index. If None, returns all edges in graph.
                  If list, uses first element.

        Returns:
            List of tuples (source, target) representing edges.
        """
        if isinstance(srce, list):
            srce = srce[0]
        if srce is None:
            return [(k, i) for k, subdict in self._adj.items() for i in subdict]
        if srce in self._adj:
            return [(srce, i) for i in self._adj[srce]]
        return []


class BinaryAcceptance(Graph):
    """
    Binary Acceptance Matrix representation with weighted graph structure.

    Converts correlation/relation matrices into boolean acceptance matrices where
    True indicates an acceptable pairwise relation (below threshold). Builds a
    weighted directed graph with a dummy target node to enable path finding algorithms.

    Attributes:
        source: Current source node index for pathfinding (default 0).
        bin_acc: Boolean matrix indicating acceptable pairwise relations.
        weights: Array of weights for each node (length N+1 includes dummy target).
        labels: Optional labels for nodes.
        dim: Dimension of the matrix (number of nodes excluding dummy target).
    """

    def __init__(
            self,
            matrix: Union[Array2D_Bool, Array2D_Float],
            weights: Optional[Union[List[float], np.ndarray]] = None,
            labels: Optional[List[str]] = None,
            threshold: Optional[float] = None,
            allow_negative_weights: bool = False
    ) -> None:
        """
        Initialise Binary Acceptance Matrix from correlation/relation matrix.

        Args:
            matrix: Square 2D matrix of boolean or float values. If float, must provide threshold.
                   True/values below threshold indicate acceptable pairwise relations.
            weights: Optional weights for each node. If None, uniform weights of 1.
                    Can be list or numpy array.
            labels: Optional string labels for nodes.
            threshold: Required if matrix is float type. Values where |matrix[i,j]| < threshold
                      are considered acceptable relations.
            allow_negative_weights: If False, raises exception for negative weights (required
                                   for subset exclusion guarantees). Default False.

        Returns:
            None

        Raises:
            ValueError: If matrix is not 2D, not square, or float type without threshold.
            Exception: If negative weights provided and allow_negative_weights is False.
        """
        super().__init__()

        self.source = 0
        self.bin_acc = self.set_binary_acceptance(matrix, threshold)
        self.weights = self.set_weights(weights, self.dim)
        self.labels = labels
        if not allow_negative_weights and min(self.weights) < 0.0:
            raise ValueError('Negative weights provided. ' +
                             'Rescale to positive or set allow_negative_weights=True')
        self.set_weighted_graph()

    @property
    def dim(self) -> int:
        """Get dimension of binary acceptance matrix (number of nodes)."""
        return self.bin_acc.shape[0]

    @property
    def get_source_row(self) -> Array1D_bool:
        """Get boolean array of acceptable relations from current source node."""
        return self.bin_acc[self.source, :]

    @property
    def get_source_row_index(self) -> Array1D_int:
        """Get indices of nodes with acceptable relations to current source."""
        return np.where(self.get_source_row)[0][self.source + 1::]

    @staticmethod
    def set_binary_acceptance(
            matrix: np.ndarray,
            threshold: Optional[float] = None
    ) -> Array2D_Bool:
        """
        Convert input matrix to boolean acceptance matrix.

        Args:
            matrix: Square 2D matrix of boolean, integer, or float values.
            threshold: Required for float matrices. Absolute values below threshold
                      become True (acceptable).

        Returns:
            Boolean matrix where True indicates acceptable pairwise relation.

        Raises:
            ValueError: If matrix is not 2D, not square, or float without threshold.
        """
        if matrix.ndim != 2:
            raise ValueError('Binary acceptance is not a 2d array')

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Binary acceptance is not square')

        if matrix.dtype == bool:
            return matrix
        elif matrix.dtype == 'int' and threshold is None:
            print(f'{10*"#"} Warning! {10*"#"}\nBinary acceptance '
                  'is array of integers, converting format to True/False')
            return np.array(abs(matrix), dtype=bool)
        elif threshold is not None:
            return abs(matrix) < threshold
        else:
            raise ValueError('Binary acceptance is not Boolean type! Convert or provide threshold')

    @staticmethod
    def set_weights(weights: Optional[Union[List[float], np.ndarray]], size: int) -> Array1D_Float:
        """
        Create weights array with dummy target weight appended.

        Args:
            weights: Optional weights for nodes. If None, creates uniform weights of 1.
            size: Number of nodes (dimension of matrix).

        Returns:
            Array of length size+1 with 0.0 appended for dummy target node.
        """
        if weights is None:
            weights = [1.0] * size
        else:
            weights = list(weights)
        if len(weights) == size:
            weights += [0.0]
        return np.array(weights)

    @staticmethod
    def set_dummy_target(bin_acc: Array2D_Bool, source: int) -> Array2D_Bool:
        """
        Extend binary acceptance matrix with dummy target node.

        Creates an (N+1) x (N+1) matrix where the new row/column represents
        the dummy target. Sets source diagonal to True and filters paths
        based on source node acceptances for optimisation.

        Args:
            bin_acc: Original N x N boolean acceptance matrix.
            source: Source node index for current search.

        Returns:
            Extended (N+1) x (N+1) boolean matrix with dummy target node.
        """
        dim = np.array(bin_acc.shape)
        new_dim = int(dim[0] + 1)
        bin_acc_plus: Array2D_Bool = np.ones((new_dim, new_dim), dtype=bool)
        bin_acc_plus[:-1:, :-1:] = bin_acc
        bin_acc_plus[source, source] = True
        # These lines make things fast!
        sub = ~bin_acc_plus[source, :]
        bin_acc_plus[:, sub] = False
        bin_acc_plus[sub, :] = False
        return bin_acc_plus

    def get_full_triu(self) -> np.ndarray:
        """
        Get upper triangle of binary acceptance matrix with dummy target.

        Returns:
            Upper triangular portion of extended matrix (includes dummy target node).
        """
        full_graph_mat = self.set_dummy_target(self.bin_acc, self.source)
        return np.triu(full_graph_mat, 1)

    def set_weighted_graph(self) -> None:
        """
        Build weighted graph from binary acceptance matrix.

        Creates directed edges with weights from acceptance matrix. Clears
        existing graph and rebuilds from current binary acceptance state.

        Returns:
            None
        """
        edges = [(ij[0], ij[1], float(self.weights[ij[1]])) for ij in np.argwhere(self.get_full_triu())]
        self.add_weighted_edges(edges, clear=True)

    def get_weight(self, path: List[int]) -> float:
        """
        Calculate total weight for a path.

        Args:
            path: List of node indices forming a path.

        Returns:
            Sum of weights for all nodes in path. Returns 0.0 for empty path.
        """
        if path:
            return float(np.sum(self.weights[path]))
        return 0.0

    def reset_source(self, source: int = 0) -> None:
        """
        Change source node and rebuild graph accordingly.

        Args:
            source: New source node index. Must be < dim. If out of range, defaults to 0.

        Returns:
            None
        """
        if source >= self.dim:
            print('Source out of range. Defaulting to zero')
            source = 0
        self.source = source
        self.set_weighted_graph()

    def sort_bam_by_weight(self) -> Array1D_int:
        """
        Sort matrix rows/columns by descending weight order.

        Reorders the binary acceptance matrix and weights array so nodes with
        higher weights appear first. Critical for WHDFS optimisation.
        Returns index mapping to convert sorted results back to original order.

        Returns:
            Index mapping array where index_map[sorted_index] = original_index.
            Use with Results.remap_path() to convert results to original indices.

        Example:
            >>> bam = BinaryAcceptance(matrix, weights=[0.1, 0.9, 0.5])
            >>> index_map = bam.sort_bam_by_weight()
            >>> # Now bam is sorted [0.9, 0.5, 0.1] = original indices [1, 2, 0]
            >>> results.remap_path(index_map)  # Convert back to original indices
        """
        index_map = np.argsort(self.weights[:-1:])[::-1].astype(int)
        self.weights[:-1:] = self.weights[index_map]
        self.bin_acc = self.bin_acc[index_map, :][:, index_map]
        if self.labels is not None:
            self.labels = [self.labels[i] for i in index_map]
        return index_map
