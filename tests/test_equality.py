"""
Tests for __eq__ methods in BinaryAcceptance, HDFS, and WHDFS classes.
"""
import numpy as np
import pytest
from pathfinder.matrix_handler import BinaryAcceptance
from pathfinder.dfs import HDFS, WHDFS


def test_binary_acceptance_equality_same():
    """Test that two BinaryAcceptance objects with same attributes are equal"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])
    weights = [1.0, 2.0, 3.0]
    labels = ['A', 'B', 'C']

    bam1 = BinaryAcceptance(matrix, weights=weights, labels=labels)
    bam2 = BinaryAcceptance(matrix, weights=weights, labels=labels)

    assert bam1 == bam2


def test_binary_acceptance_equality_different_source():
    """Test that BinaryAcceptance objects with different sources are not equal"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])
    weights = [1.0, 2.0, 3.0]

    bam1 = BinaryAcceptance(matrix, weights=weights)
    bam2 = BinaryAcceptance(matrix, weights=weights)
    bam2.reset_source(1)

    assert bam1 != bam2


def test_binary_acceptance_equality_different_matrix():
    """Test that BinaryAcceptance objects with different matrices are not equal"""
    matrix1 = np.array([[True, False], [False, True]])
    matrix2 = np.array([[False, True], [True, False]])
    weights = [1.0, 2.0]

    bam1 = BinaryAcceptance(matrix1, weights=weights)
    bam2 = BinaryAcceptance(matrix2, weights=weights)

    assert bam1 != bam2


def test_binary_acceptance_equality_different_weights():
    """Test that BinaryAcceptance objects with different weights are not equal"""
    matrix = np.array([[True, False], [False, True]])

    bam1 = BinaryAcceptance(matrix, weights=[1.0, 2.0])
    bam2 = BinaryAcceptance(matrix, weights=[2.0, 3.0])

    assert bam1 != bam2


def test_binary_acceptance_equality_different_labels():
    """Test that BinaryAcceptance objects with different labels are not equal"""
    matrix = np.array([[True, False], [False, True]])
    weights = [1.0, 2.0]

    bam1 = BinaryAcceptance(matrix, weights=weights, labels=['A', 'B'])
    bam2 = BinaryAcceptance(matrix, weights=weights, labels=['C', 'D'])

    assert bam1 != bam2


def test_binary_acceptance_equality_one_with_labels_one_without():
    """Test that BinaryAcceptance objects where one has labels and other doesn't are not equal"""
    matrix = np.array([[True, False], [False, True]])
    weights = [1.0, 2.0]

    bam1 = BinaryAcceptance(matrix, weights=weights, labels=['A', 'B'])
    bam2 = BinaryAcceptance(matrix, weights=weights, labels=None)

    assert bam1 != bam2


def test_binary_acceptance_equality_with_index_map():
    """Test that BinaryAcceptance objects with same index_map are equal"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])
    weights = [3.0, 1.0, 2.0]

    bam1 = BinaryAcceptance(matrix, weights=weights)
    bam1.sort_bam_by_weight()

    bam2 = BinaryAcceptance(matrix, weights=weights)
    bam2.sort_bam_by_weight()

    assert bam1 == bam2


def test_binary_acceptance_equality_different_index_map():
    """Test that BinaryAcceptance objects with different index_map states are not equal"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])
    weights = [3.0, 1.0, 2.0]

    bam1 = BinaryAcceptance(matrix, weights=weights)
    bam1.sort_bam_by_weight()

    bam2 = BinaryAcceptance(matrix, weights=weights)
    # bam2 not sorted, so no index_map

    assert bam1 != bam2


def test_binary_acceptance_equality_not_implemented_for_other_types():
    """Test that comparing BinaryAcceptance with other types returns NotImplemented"""
    matrix = np.array([[True, False], [False, True]])
    bam = BinaryAcceptance(matrix)

    assert bam != "not a BinaryAcceptance"
    assert bam != 42
    assert bam is not None


def test_hdfs_equality_same():
    """Test that two HDFS objects with same attributes are equal"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])
    weights = [1.0, 2.0, 3.0]

    bam = BinaryAcceptance(matrix, weights=weights)
    hdfs1 = HDFS(bam, top=5, allow_subset=False)
    hdfs2 = HDFS(bam, top=5, allow_subset=False)

    assert hdfs1 == hdfs2


def test_hdfs_equality_different_bam():
    """Test that HDFS objects with different BAM objects are not equal"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])

    bam1 = BinaryAcceptance(matrix, weights=[1.0, 2.0, 3.0])
    bam2 = BinaryAcceptance(matrix, weights=[3.0, 2.0, 1.0])

    hdfs1 = HDFS(bam1, top=5)
    hdfs2 = HDFS(bam2, top=5)

    assert hdfs1 != hdfs2


def test_hdfs_equality_different_weight_func():
    """Test that HDFS objects with different weight functions are not equal"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])

    bam = BinaryAcceptance(matrix, weights=[1.0, 2.0, 3.0])
    hdfs1 = HDFS(bam, top=5)
    hdfs2 = HDFS(bam, top=5)

    # Change weight function
    hdfs2.weight_func = lambda x: sum(x)

    assert hdfs1 != hdfs2


def test_hdfs_equality_not_implemented_for_other_types():
    """Test that comparing HDFS with other types returns NotImplemented"""
    matrix = np.array([[True, False], [False, True]])
    bam = BinaryAcceptance(matrix)
    hdfs = HDFS(bam, top=5)

    assert hdfs != "not an HDFS"
    assert hdfs != 42


@pytest.mark.filterwarnings("ignore:WHDFS performance:UserWarning")
def test_whdfs_equality_same():
    """Test that two WHDFS objects with same attributes are equal"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])
    weights = [1.0, 2.0, 3.0]

    bam = BinaryAcceptance(matrix, weights=weights)
    whdfs1 = WHDFS(bam, top=5, allow_subset=False, auto_sort=False)
    whdfs2 = WHDFS(bam, top=5, allow_subset=False, auto_sort=False)

    assert whdfs1 == whdfs2


@pytest.mark.filterwarnings("ignore:WHDFS performance:UserWarning")
def test_whdfs_equality_different_bam():
    """Test that WHDFS objects with different BAM objects are not equal"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])

    bam1 = BinaryAcceptance(matrix, weights=[1.0, 2.0, 3.0])
    bam2 = BinaryAcceptance(matrix, weights=[3.0, 2.0, 1.0])

    whdfs1 = WHDFS(bam1, top=5, auto_sort=False)
    whdfs2 = WHDFS(bam2, top=5, auto_sort=False)

    assert whdfs1 != whdfs2


@pytest.mark.filterwarnings("ignore:WHDFS performance:UserWarning")
def test_whdfs_equality_different_weight_func():
    """Test that WHDFS objects with different weight functions are not equal"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])

    bam = BinaryAcceptance(matrix, weights=[1.0, 2.0, 3.0])
    whdfs1 = WHDFS(bam, top=5, auto_sort=False)
    whdfs2 = WHDFS(bam, top=5, auto_sort=False)

    # Change weight function
    whdfs2.weight_func = lambda x: sum(x)

    assert whdfs1 != whdfs2


@pytest.mark.filterwarnings("ignore:WHDFS performance:UserWarning")
def test_whdfs_equality_different_wlimit_func():
    """Test that WHDFS objects with different wlimit functions are not equal"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])

    bam = BinaryAcceptance(matrix, weights=[1.0, 2.0, 3.0])
    whdfs1 = WHDFS(bam, top=5, auto_sort=False)
    whdfs2 = WHDFS(bam, top=5, auto_sort=False)

    # Change wlimit function
    whdfs2.wlimit_func = lambda x: sum(x)

    assert whdfs1 != whdfs2


@pytest.mark.filterwarnings("ignore:WHDFS performance:UserWarning")
def test_whdfs_equality_different_top_weights():
    """Test that WHDFS objects with different top_weights functions are not equal"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])

    bam = BinaryAcceptance(matrix, weights=[1.0, 2.0, 3.0])
    whdfs1 = WHDFS(bam, top=5, auto_sort=False)
    whdfs2 = WHDFS(bam, top=5, auto_sort=False)

    # Change top_weights function
    whdfs2.set_top_weight(lambda: 100.0)

    assert whdfs1 != whdfs2


@pytest.mark.filterwarnings("ignore:WHDFS performance:UserWarning")
def test_whdfs_equality_not_implemented_for_other_types():
    """Test that comparing WHDFS with other types returns NotImplemented"""
    matrix = np.array([[True, False], [False, True]])
    bam = BinaryAcceptance(matrix)
    whdfs = WHDFS(bam, top=5, auto_sort=False)

    assert whdfs != "not a WHDFS"
    assert whdfs != 42


@pytest.mark.filterwarnings("ignore:WHDFS performance:UserWarning")
def test_hdfs_whdfs_not_equal():
    """Test that HDFS and WHDFS objects are not equal even with same BAM"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])

    bam = BinaryAcceptance(matrix, weights=[1.0, 2.0, 3.0])
    hdfs = HDFS(bam, top=5)
    whdfs = WHDFS(bam, top=5, auto_sort=False)

    # These are different types, so should not be equal
    assert hdfs != whdfs


def test_whdfs_equality_with_auto_sort():
    """Test that WHDFS objects are equal when both use auto_sort"""
    matrix = np.array([[True, False, True],
                       [False, True, False],
                       [True, False, True]])
    weights = [1.0, 2.0, 3.0]

    # Create separate BAMs so they get sorted independently
    bam1 = BinaryAcceptance(matrix.copy(), weights=weights.copy())
    bam2 = BinaryAcceptance(matrix.copy(), weights=weights.copy())

    whdfs1 = WHDFS(bam1, top=5, allow_subset=False, auto_sort=True)
    whdfs2 = WHDFS(bam2, top=5, allow_subset=False, auto_sort=True)

    # Both should be sorted identically
    assert whdfs1 == whdfs2
