import numpy as np
from pathfinder.matrix_handler import BinaryAcceptance, Graph
import pytest

np.random.seed(1)


def get_basic_data(ret_dtype: object = bool):
    matrix = [[0, 1, 1, 0],
              [1, 0, 1, 0],
              [1, 1, 0, 1],
              [0, 0, 1, 0]]
    return np.array(matrix, dtype=ret_dtype)


def test_binaryacceptance():
    bam1 = BinaryAcceptance(get_basic_data(ret_dtype=bool))
    bam2 = BinaryAcceptance(get_basic_data(ret_dtype=int))
    assert not all(bam1.bin_acc.diagonal())
    assert not all(bam2.bin_acc.diagonal())
    assert all([list(item1) == list(item2) for item1, item2 in zip(bam1.bin_acc, bam2.bin_acc)])
    # check error is raised when providing floats
    with pytest.raises(ValueError):
        BinaryAcceptance(get_basic_data(ret_dtype=float))


def test_threshold():
    # invert basic matrix and convert to float
    inv_matrix = np.array(~get_basic_data(ret_dtype=bool), dtype=float)
    # initiate bam with threshold of ij < 0.5 == True
    bam1 = BinaryAcceptance(np.array(inv_matrix, dtype=float), threshold=0.5)
    bam2 = BinaryAcceptance(get_basic_data(ret_dtype=bool))
    # check bam is consistent with basic data dtype = bool
    assert all([list(item1) == list(item2) for item1, item2 in zip(bam1.bin_acc, bam2.bin_acc)])


def test_getters():
    weights = np.array([1, 1, 1, -3.0])
    bam = BinaryAcceptance(get_basic_data(ret_dtype=bool), weights=weights, allow_negative_weights=True)
    assert bam.get_weight([1, 2, 3]) == -1
    assert all([sum(item) == num for item, num in zip(bam.get_full_triu(), [3, 2, 1, 0, 0])])


def test_reset_source():
    bam = BinaryAcceptance(get_basic_data(ret_dtype=bool))
    assert bam.source == 0
    bam.reset_source(1)
    assert bam.source == 1
    bam.reset_source(10)
    assert bam.source == 0


def test_graph():
    matrix = get_basic_data(ret_dtype=bool)
    weights = np.ones(len(matrix))
    edges = [(*ij, weights[ij[1]]) for ij in np.argwhere(np.triu(matrix, 1))]
    graph = Graph()
    graph.add_weighted_edges(edges)
    assert graph.edges() == [(0, 1), (0, 2), (1, 2), (2, 3)]
    assert graph.edges(1) == [(1, 2)]
    assert graph.edges(3) == []


if __name__ == '__main__':
    test_reset_source()
