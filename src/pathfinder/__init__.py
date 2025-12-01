#!/usr/bin/env python3
"""
#####################################
# PATH FINDER MODULE                #
# Author J.Yellen                   #
#####################################
"""

from .matrix_handler import BinaryAcceptance
from .dfs import HDFS, WHDFS
from .result import Results
from typing import Union
import numpy as np

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = [
    'BinaryAcceptance',
    'HDFS',
    'WHDFS',
    'Results',
    'find_best_combinations',
    '__version__'
]


def find_best_combinations(matrix, weights=None, threshold=None, top=10,
                           allow_subset=False, runs=None, labels=None,
                           algorithm='whdfs', verbose=False) -> Union[HDFS, WHDFS]:
    """
    High-level convenience function to find optimal feature combinations.

    This function wraps the complete PathFinder workflow: creating a BinaryAcceptance
    matrix, running the search algorithm, and returning results automatically remapped
    to original indices.

    Arguments:
        matrix: Square 2D array of pairwise relations (boolean or float)
        weights: Optional array of feature weights (default: uniform weights)
        threshold: Required for float matrices - values below threshold are compatible
        top: Number of top results to return (default: 10)
        allow_subset: Allow paths that are subsets of longer paths (default: False)
        runs: Number of source nodes to search from (default: all for HDFS, all for WHDFS)
        labels: Optional feature labels for results
        algorithm: 'whdfs' (default, faster) or 'hdfs' (exhaustive)
        verbose: Print results to console (default: False)

    Returns:
        Results: Results object with paths automatically remapped to original indices

    Example:
        >>> import numpy as np
        >>> correlation = np.random.rand(20, 20)
        >>> weights = np.random.rand(20)
        >>> results = find_best_combinations(
        ...     matrix=correlation,
        ...     weights=weights,
        ...     threshold=0.7,
        ...     top=5
        ... )
        >>> print(results.get_paths[0])  # Best combination
    """
    # Create BinaryAcceptance object
    bam = BinaryAcceptance(
        matrix=matrix,
        weights=weights,
        labels=labels,
        threshold=threshold
    )

    # Select and run algorithm
    if algorithm.lower() == 'hdfs':
        searcher = HDFS(bam, top=top, allow_subset=allow_subset)
    elif algorithm.lower() == 'whdfs':
        # WHDFS auto-sorts by default for optimal performance
        searcher = WHDFS(bam, top=top, allow_subset=allow_subset, auto_sort=True)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose 'hdfs' or 'whdfs'.")

    # Find paths
    searcher.find_paths(runs=runs, verbose=verbose)

    # For WHDFS, results are automatically remapped via property
    # For HDFS, we need manual remapping if we sorted
    if algorithm.lower() == 'hdfs':
        return searcher
    else:
        # WHDFS.get_paths already handles remapping
        return searcher
