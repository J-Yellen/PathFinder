"""
#####################################
# Part of the PATH FINDER  Module   #
# Author J.Yellen                   #
#####################################
"""
from functools import partial
from itertools import islice
import numpy as np
from .matrix_handler import BinaryAcceptance
from .result import Results
from typing import Iterable, Iterator, Optional, Tuple, Callable


class HDFS(Results):

    def __init__(self, binary_acceptance_obj: BinaryAcceptance, top: int = 10,
                 allow_subset: bool = False) -> None:
        """
            Hereditary Depth First Search Class
        Args:
            binary_acceptance_obj (BinaryAcceptance): BinaryAcceptance Object containing
            top (int, optional): _description_. Defaults to 10.
            allow_subset (bool, optional): _description_. Defaults to True.
        """
        if not allow_subset and min(binary_acceptance_obj.weights) < 0:
            raise Exception('WARNING! Negative weights provided. Subset exclusion cannot be guarantied!')
        super().__init__(paths=[{}], weights=[0.0], top=top, allow_subset=allow_subset)
        self.bam = binary_acceptance_obj
        self.weight_func = self.bam.get_weight
        self.n_iteration = 0

    def hdfs(self, trim: bool = True, ignore_child: Optional[list] = None) -> Iterator:
        """
        Hereditary Depth First Search
        Returns all paths under the Hereditary condition.
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
        """Break *iterable* into lists of length *n*: """
        def take(n, iterable):
            return list(islice(iterable, n))
        return iter(partial(take, n, iter(iterable)), [])

    def find_paths(self, runs: Optional[int] = None, source_node: int = 0,
                   ignore_child: Optional[Tuple[list, int]] = None, verbose: bool = False) -> None:
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
            super().__init__(paths=[{}], weights=[0.0], top=self.top, allow_subset=self.allow_subset)

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


class WHDFS(Results):

    def __init__(self, binary_acceptance_obj: BinaryAcceptance, top: int = 10, allow_subset: bool = False) -> None:
        """
        Weighted Hereditary Depth First Search
        """
        if not allow_subset and min(binary_acceptance_obj.weights) < 0:
            raise Exception('WARNING! Negative weights provided. Subset exclusion cannot be guarantied!')
        super(WHDFS, self).__init__(paths=[{}], weights=[0.0], top=top, allow_subset=allow_subset)

        self.bam = binary_acceptance_obj
        self.weight_func = self.bam.get_weight
        # self.wlimit_func = self.bam.get_weight
        self.wlimit_func = self.weight_func
        self.top_weight = self._top_weights_default
        self.shared_memory = False
        self.n_iteration = 0

    @property
    def weight_func(self) -> float:
        return self._weight_func

    @weight_func.setter
    def weight_func(self, weight_function: Callable) -> None:
        self._weight_func = weight_function

    @property
    def wlimit_func(self) -> float:
        return self._wlimit_func

    @wlimit_func.setter
    def wlimit_func(self, weight_function: Callable) -> None:
        self._wlimit_func = weight_function

    def top_weight(self) -> float:
        return self._top_weights

    def set_top_weight(self, weight_function: Callable) -> None:
        self._top_weights = weight_function

    def _top_weights_default(self) -> float:
        return min(self.get_weights)

    def shared_memory_update(self, value) -> None:
        if not self.shared_memory:
            raise AttributeError('Shared_memory_update not set, please use set_shared_memory_update.')
        self._shared_memory_update(value)

    def set_shared_memory_update(self, shared_memory_function: Callable) -> None:
        self.shared_memory = True
        self._shared_memory_update = shared_memory_function

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
                        if self.shared_memory:
                            max_wgt = self.shared_memory_update(max_wgt)
                        else:
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
            super().__init__(paths=[{}], weights=[0.0], top=self.top, allow_subset=self.allow_subset)
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
