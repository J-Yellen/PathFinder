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
        if len(self.res) > 1:
            super().__init__(paths=[[]], weights=[0.0], top=self.top)

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
        self.wlimit_func = self.bam.get_weight

    def whdfs(self) -> None:
        """
        Weighted Hereditary Depth First Search
        Returns best path for a given source under
        the weighted Hereditary condition.
        """
        cutoff = self.bam.dim + 1
        target = self.bam.dim
        visited = dict.fromkeys([self.bam.source])                          # initiate the visited list with the source node
        stack = [(v for _, v in self.bam.edges(self.bam.source))]           # list of generators that builds to provide the subset of available nodes for each child with all nodes > child
        good_nodes = [set(v for _, v in self.bam.edges(self.bam.source))]   # compleat set of available nodes for each child
        max_wgt = np.array(self.get_weights)                                # get current max weight
        while stack:                                                        # iterate over nodes building and dropping from stack until empty
            children = stack[-1]                                            # define children as the generator from the last element of stack
            child = next(children, None)                                    # The child node is the next element from children
            if child is None:                                               # if no child drop last elements from stack, good nodes and visited
                stack.pop()
                good_nodes.pop()
                visited.popitem()
            elif len(visited) < cutoff:                                             # number of nodes in path less then the length of the correlations
                if child in visited:                                                # ensure no repeated nodes
                    continue
                pth = list(visited) + [child]                                               # define current path bing considered
                gn = set(v for _, v in self.bam.edges(child)).intersection(good_nodes[-1])  # Intersection of nodes available to the child with those available to all previous nodes in path
                child_pths = np.array(list(gn))                                             # list the available nodes from the set gn
                currnt_wgt = self.weight_func(pth)                                      # weight of current path
                remain_wgt = self.wlimit_func(list(child_pths[(child_pths > child)]))   # upper limit on the weight available to the child
                if child == target:
                    if currnt_wgt > max_wgt.min():
                        self.add_res(pth[:-1:], currnt_wgt)                                         # update result
                        max_wgt = np.array(self.get_weights)
                if (currnt_wgt + remain_wgt) > max_wgt.min():                                       # is the remaining weight enough to continue "down this route"
                    visited[child] = None
                    if target not in visited:
                        good_nodes.append(gn)                                                       # add gn to good nodes
                        stack.append((v for _, v in self.bam.edges(child) if v in good_nodes[-1]))  # add the nest node generator to stack
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
        if runs is None or runs > self.bam.dim:
            runs = self.bam.dim

        for i in range(0, runs):
            self.whdfs()
            if i < self.bam.dim-1:
                self.bam.reset_source(i+1)
        self.bam.reset_source()
        if verbose:
            print(self)
