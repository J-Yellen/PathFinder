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

class HDFS(Results):

    def __init__(self, binary_acceptance_obj:BinaryAcceptance, top:int=10, ignore_subset:bool=True)-> None:
        """
        Hereditary Depth First Search
        """
        super().__init__(paths=[{}], weights=[0.0], top=top, ignore_subset=ignore_subset)
        self.bam = binary_acceptance_obj
        self.weight_func = self.bam.get_weight

    def hdfs(self, trim:bool=True) -> list:
        """
        Hereditary Depth First Search
        Returns all paths under the Hereditary condition.
        """
        target = self.bam.dim
        cutoff = self.bam.dim + 1
        visited = dict.fromkeys([self.bam.source])
        stack = [(v for _, v in self.bam.edges(self.bam.source))]
        good_nodes = [set(v for _, v in self.bam.edges(self.bam.source))]
        while stack:
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
    def chunked(iterable, n):
        """Break *iterable* into lists of length *n*:"""
        def take(n, iterable):
            return list(islice(iterable, n))
        return iter(partial(take, n, iter(iterable)), [])

    def find_paths(self, runs:int=None, verbose=False) -> None:
        """
        Evaluate the available paths/subsets
        runs: number of initial nodes starting from 0
        """
        self.bam.reset_source()
        if len(self.res) > 1:
            super().__init__(paths=[[]], weights=[-np.inf], top=self.top, ignore_subset=self.ignore_subset)

        if runs is None or runs > self.bam.dim:
            runs = self.bam.dim
        for i in range(0, runs):
            all_p = self.hdfs()
            for item in self.chunked(all_p, 500):
                paths = list(item)
                weights = [self.weight_func(p) for p in paths if p]
                self.bulk_add(paths, weights)
                self
            if i < self.bam.dim-1:
                self.bam.reset_source(i+1)
        self.bam.reset_source()
        if verbose:
            print(self)

class WHDFS(Results):

    def __init__(self, binary_acceptance_obj:BinaryAcceptance, top:int=10, ignore_subset:bool=True) -> None:
        """
        Weighted Hereditary Depth First Search
        """
        super().__init__(paths=[{}], weights=[0.0], top=top, ignore_subset=ignore_subset)
        self.bam = binary_acceptance_obj
        self.weight_func = self.bam.get_weight
        self.wlimit_func = self.bam.get_weight_lim if ignore_subset else self.bam.get_abs_weight_lim

    def whdfs(self) -> None:
        """
        Weighted Hereditary Depth First Search
        Returns best path for a given source under
        the weighted Hereditary condition.
        """
        cutoff = self.bam.dim + 1
        target = self.bam.dim
        # initiate the visited list with the source node
        visited = dict.fromkeys([self.bam.source])
        # list of generators that builds to provide the subset of available nodes for each child with all nodes > child
        stack = [(v for _, v in self.bam.edges(self.bam.source))]
        # compleat set of available nodes for each child
        good_nodes = [set(v for _, v in self.bam.edges(self.bam.source))]
        # get current max weight
        max_wgt = np.array(self.get_weights)
        # iterate over nodes building and dropping from stack until empty
        while stack:
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
                    if currnt_wgt > max_wgt.min():
                        # update result
                        self.add_res(pth[:-1:], currnt_wgt)
                        max_wgt = np.array(self.get_weights)
                # is the remaining weight enough to continue "down this route"
                if (currnt_wgt + remain_wgt) > max_wgt.min():
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

    def find_paths(self, runs:int=None, verbose:bool=False) -> None:
        """
        Evaluate the available paths/subsets
        runs: number of initial nodes starting from 0
        """
        self.bam.reset_source()
        if len(self.res) > 1:
            super().__init__(paths=[[]], weights=[-np.inf], top=self.top, ignore_subset=self.ignore_subset)
        if runs is None or runs > self.bam.dim:
            runs = self.bam.dim

        for i in range(0, runs):
            self.whdfs()
            if i < self.bam.dim-1:
                self.bam.reset_source(i+1)
        self.bam.reset_source()
        if verbose:
            print(self)
