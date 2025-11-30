"""
#####################################
# Part of the PATH FINDER  Module   #
# Author J.Yellen                   #
#####################################
"""
from functools import partial
from itertools import islice
import numpy as np
from numpy import ndarray
from .matrix_handler import BinaryAcceptance
from .result import Results
from typing import Iterable, Iterator, List, Optional, Callable, Union, Dict


class HDFS(Results):

    def __init__(self, binary_acceptance_obj: BinaryAcceptance, top: Optional[int] = 10,
                 allow_subset: bool = False) -> None:
        """
            Hereditary Depth First Search Class
        Arguments:
            binary_acceptance_obj (BinaryAcceptance): BinaryAcceptance Object containing
            top (int, optional): _description_. Defaults to 10.
            allow_subset (bool, optional): _description_. Defaults to True.
        """
        if not allow_subset and min(binary_acceptance_obj.weights) < 0:
            raise Exception('WARNING! Negative weights provided. Subset exclusion cannot be guarantied!')
        super().__init__(top=top, allow_subset=allow_subset)
        self.bam = binary_acceptance_obj
        self.weight_func = self.bam.get_weight
        self.n_iteration = 0

    def hdfs(self, trim: bool = True, ignore_child: Optional[List] = None) -> Iterator:
        """
        Hereditary Depth First Search. Yields all paths from source to target node
        Arguments:
            trim (bool, optional): If True, yield paths without the target node. Defaults to True.
            ignore_child (Optional[List], optional): List of child nodes to ignore. Defaults to None.
        Yields:
            Iterator: Yields paths as lists of node indices.
        """
        if ignore_child is None:
            ignore_child = []
        target = self.bam.dim
        cutoff = self.bam.dim + 1
        visited = dict.fromkeys([self.bam.source])
        stack = [(v for _, v in self.bam.edges(self.bam.source) if v not in ignore_child)]
        good_nodes = [set(v for _, v in self.bam.edges(self.bam.source))]
        while stack:
            self.n_iteration += 1
            children = stack[-1]
            child = next(children, None)
            if child is None:
                stack.pop()
                good_nodes.pop()
                visited.popitem()
            elif len(visited) < cutoff:
                if child in visited:
                    continue
                if child == target:
                    if trim:
                        yield list(visited)
                    else:
                        yield list(visited) + [child]
                visited[child] = None
                if target not in visited:
                    good_children = set(v for _, v in self.bam.edges(child))
                    good_nodes += [good_children.intersection(good_nodes[-1])]
                    stack.append((v for _, v in self.bam.edges(child) if v in good_nodes[-1]))
                else:
                    visited.popitem()
            else:  # len(visited) == cutoff:
                if trim:
                    yield list(visited)
                else:
                    yield list(visited) + [child]
                stack.pop()
                good_nodes.pop()
                visited.popitem()

    @staticmethod
    def chunked(iterable: Iterable, n: int) -> Iterator:
        """Break *iterable* into lists of length *n*:
         Arguments:
             iterable (Iterable): The iterable to be chunked.
             n (int): The size of each chunk.
         Returns:
             Iterator: An iterator over chunks of the iterable.
         """
        def take(n, iterable):
            return list(islice(iterable, n))
        return iter(partial(take, n, iter(iterable)), [])

    def find_paths(self, runs: Optional[int] = None, source_node: int = 0,
                   ignore_child: Optional[List] = None, verbose: bool = False) -> None:
        """
        Evaluate the available paths/subsets
        runs : number of initial nodes starting from 0
        """
        if ignore_child is None:
            ignore_child = []
        if isinstance(ignore_child, (int, float)):
            ignore_child = [int(ignore_child)]
        self.bam.reset_source(source=source_node)
        if len(self.res) > 1:
            self.n_iteration = 0
            super().__init__(top=self.top, allow_subset=self.allow_subset)

        if runs is None or runs > self.bam.dim:
            runs = self.bam.dim
        for i in range(source_node, runs + source_node):
            for item in self.chunked(self.hdfs(ignore_child=ignore_child), 500):
                paths = list(item)
                weights = [self.weight_func(p) for p in paths if p]
                self.bulk_add_result(paths, weights)
            if i < self.bam.dim - 1:
                self.bam.reset_source(i + 1)
        self.bam.reset_source()
        if verbose:
            print(self)

    def get_sorted_results(self) -> None:
        """  not implemented for HDFS """
        raise NotImplementedError("get_sorted_results is not implemented for HDFS")

    def remap_path(self, index_map: Optional[Union[Dict, List, ndarray]] = None,
                   weight_offset: float = 0.0) -> 'Results':
        """
        Remap paths to original indices. If index_map is not provided,
        uses the index_map stored in self.bam (if available).

        Args:
            index_map: Optional index mapping. If None, uses self.bam._index_map.
            weight_offset: Optional weight offset to subtract (per path element).

        Returns:
            Results object with remapped paths.
        """
        if index_map is None:
            index_map = self.bam._index_map
        return super().remap_path(index_map, weight_offset)


class WHDFS(Results):

    def __init__(self, binary_acceptance_obj: BinaryAcceptance, top: int = 10,
                 allow_subset: bool = False, auto_sort: bool = True) -> None:
        """
        Weighted Hereditary Depth First Search

        Arguments:
            binary_acceptance_obj (BinaryAcceptance): BinaryAcceptance Object
            top (int, optional): Number of top results to retain. Defaults to 10.
            allow_subset (bool, optional): Allow paths that are subsets of other paths. Defaults to False.
            auto_sort (bool, optional): Automatically sort BAM by weights for optimal performance. Defaults to True.
        """
        if not allow_subset and min(binary_acceptance_obj.weights) < 0:
            raise Exception('WARNING! Negative weights provided. Subset exclusion cannot be guarantied!')
        super(WHDFS, self).__init__(top=top, allow_subset=allow_subset)

        self.bam = binary_acceptance_obj

        # Auto-sort for optimal performance
        if auto_sort:
            self.bam.sort_bam_by_weight()
        else:
            # Warn if weights are not uniform (performance will suffer)
            weights = self.bam.weights
            if len(set(weights)) > 1:  # Non-uniform weights
                import warnings
                warnings.warn(
                    "WHDFS performance is significantly degraded without weight-based sorting. "
                    "Consider using auto_sort=True (default) for up to 1000× speedup.",
                    UserWarning
                )

        self.weight_func = self.bam.get_weight
        self.wlimit_func = self.bam.get_weight
        self._top_weights = self._top_weights_default
        self.n_iteration = 0

    @property
    def weight_func(self) -> Callable:
        return self._weight_func

    @weight_func.setter
    def weight_func(self, weight_function: Callable) -> None:
        self._weight_func = weight_function

    @property
    def wlimit_func(self) -> Callable:
        return self._wlimit_func

    @wlimit_func.setter
    def wlimit_func(self, weight_function: Callable) -> None:
        self._wlimit_func = weight_function

    def top_weight(self) -> float:
        return self._top_weights()

    def set_top_weight(self, weight_function: Callable) -> None:
        self._top_weights = weight_function

    def _top_weights_default(self) -> float:
        # Return minimum weight threshold for pruning
        # Only prune if we've found at least 'top' paths
        if self._top is None or len(self._res) < self._top:
            return float('-inf')  # Accept any path until we have 'top' paths
        return min(self.get_weights)

    def whdfs(self, ignore_child: Optional[list] = None) -> None:
        """
        Weighted Hereditary Depth First Search
        Returns best path for a given source under
        the weighted Hereditary condition.
        """
        if ignore_child is None:
            ignore_child = []
        cutoff = self.bam.dim + 1
        target = self.bam.dim
        # initiate the visited list with the source node
        visited = dict.fromkeys([self.bam.source])
        # list of generators that builds to provide the subset of available nodes for each child with all nodes > child
        stack = [(v for _, v in self.bam.edges(self.bam.source) if v not in ignore_child)]
        # compleat set of available nodes for each child
        good_nodes = [set(v for _, v in self.bam.edges(self.bam.source))]
        # get current max weight
        max_wgt = self.top_weight()
        # iterate over nodes building and dropping from stack until empty
        while stack:
            self.n_iteration += 1
            # define children as the generator from the last element of stack
            children = stack[-1]
            # The child node is the next element from children
            child = next(children, None)
            # if no child drop last elements from stack, good nodes and visited
            if child is None:
                stack.pop()
                good_nodes.pop()
                visited.popitem()
            # number of nodes in path less then the length of the correlations
            elif len(visited) < cutoff:
                # ensure no repeated nodes
                if child in visited:
                    continue
                # define current path bing considered
                pth = list(visited) + [child]
                # Intersection of nodes available to the child with those available to all previous nodes in path
                gn = set(v for _, v in self.bam.edges(child)).intersection(good_nodes[-1])
                # list the available nodes from the set gn
                child_pths = np.array(list(gn))
                # weight of current path
                currnt_wgt = self.weight_func(pth)
                # upper limit on the weight available to the child
                remain_wgt = self.wlimit_func(list(child_pths[(child_pths > child)]))
                if child == target:
                    if currnt_wgt > max_wgt:
                        # update result
                        self.add_result(pth[:-1:], currnt_wgt)
                        max_wgt = self.top_weight()
                # is the remaining weight enough to continue "down this route"
                if (currnt_wgt + remain_wgt) > max_wgt:
                    visited[child] = None
                    if target not in visited:
                        # add gn to good nodes
                        good_nodes.append(gn)
                        # add the nest node generator to stack
                        stack.append((v for _, v in self.bam.edges(child) if v in good_nodes[-1]))
                    else:
                        visited.popitem()
            else:  # len(visited) == cutoff:
                stack.pop()
                good_nodes.pop()
                visited.popitem()

    def find_paths(self, runs: Optional[int] = None, source_node: int = 0,
                   ignore_child: Optional[list] = None, verbose: bool = False,
                   reset_result: bool = True) -> None:
        """
        Evaluate the available paths/subsets
        runs : number of initial nodes starting from 0
        """

        if reset_result:
            super().__init__(top=self.top, allow_subset=self.allow_subset)
            self.n_iteration = 0
        if runs is None or runs > self.bam.dim:
            runs = self.bam.dim
        self.bam.reset_source(source=source_node)
        if ignore_child is None:
            ignore_child = [None] * runs
        if len(ignore_child) != runs:
            raise Exception('"ignore_child" length does not match the number of runs')
        for i in range(source_node, runs + source_node):
            ignore = ignore_child[i]
            self.whdfs(ignore_child=ignore)
            if i < self.bam.dim - 1:
                self.bam.reset_source(i + 1)
        self.bam.reset_source()
        if verbose:
            print(self)

    @property
    def get_paths(self) -> List[List[int]]:
        """Get paths with automatic remapping to original indices if auto_sort was used."""
        if self.bam._index_map is not None:
            remapped = self.remap_path(self.bam._index_map)
            return remapped.get_paths
        return super().get_paths

    @property
    def get_weights(self) -> List[float]:
        """Get weights (unchanged by remapping)."""
        return super().get_weights

    def get_sorted_paths(self) -> List[List[int]]:
        """
        Get paths in sorted index space (as stored internally).

        Use this when you need paths that match the sorted BAM matrix,
        e.g., for plotting or debugging.

        Returns:
            List of paths in sorted index space (before remapping).
        """
        return super().get_paths

    def get_sorted_results(self) -> 'Results':
        """
        Get a new Results object with paths in sorted index space.

        Returns a complete Results object (not just paths) in sorted space,
        useful for saving, further processing, or analysis.

        Returns:
            Results object with paths in sorted index space.

        Example:
            >>> whdfs = WHDFS(bam, top=5, auto_sort=True)
            >>> whdfs.find_paths()
            >>> sorted_results = whdfs.get_sorted_results()
            >>> sorted_results.to_json("sorted_results.json")
        """
        return Results.from_list_of_results(self.res)

    def remap_path(self, index_map: Optional[Union[Dict, List, ndarray]] = None,
                   weight_offset: float = 0.0) -> 'Results':
        """
        Remap paths to original indices. If index_map is not provided,
        uses the index_map stored in self.bam (if available).

        Args:
            index_map: Optional index mapping. If None, uses self.bam._index_map.
            weight_offset: Optional weight offset to subtract (per path element).

        Returns:
            Results object with remapped paths.
        """
        if index_map is None:
            index_map = self.bam._index_map
        return super().remap_path(index_map, weight_offset)

    def __str__(self) -> str:
        """
        String representation showing paths in original index space.
        When auto_sort was used, automatically remaps for display.
        """
        if self.bam._index_map is not None:
            # Show remapped paths for user-friendly output
            remapped = self.remap_path(self.bam._index_map)
            return str(remapped)
        return super().__str__()
