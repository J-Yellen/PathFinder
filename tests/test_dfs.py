import numpy as np
from pathfinder.matrix_handler import BinaryAcceptance
from pathfinder.dfs import HDFS, WHDFS


def pseudo_data(N=25, p=0.05, seed=1) -> np.ndarray:
    np.random.seed(seed)
    pseudo = np.triu(np.random.choice([True, False], size=(N, N), p=[p, 1 - p]), 1)
    pseudo += pseudo.T
    return pseudo


def pseudo_weights(N=25, sort=True, seed=1) -> np.ndarray:
    np.random.seed(seed)
    if sort:
        return np.sort(np.random.rand(N))[::-1]
    else:
        return np.random.rand(N)


def test_HDFS_1():
    pseudo = pseudo_data(N=25, p=0.1)
    weights = pseudo_weights(N=25, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights)
    index_map = bam.sort_bam_by_weight()
    hdfs = HDFS(binary_acceptance_obj=bam, top=3, allow_subset=False)
    hdfs.find_paths(verbose=False)
    assert hdfs.best.path == {1, 2}
    assert hdfs.remap_path(index_map).best.path == {13, 24}


def test_WHDFS_1():
    pseudo = pseudo_data(N=25, p=0.1)
    weights = pseudo_weights(N=25, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights)
    index_map = bam.sort_bam_by_weight()
    whdfs = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False)
    whdfs.find_paths(verbose=False)
    assert whdfs.best.path == {1, 2}
    assert whdfs.remap_path(index_map).best.path == {13, 24}


def test_WHDFS_eq_HDFS():
    pseudo = pseudo_data(N=25, p=0.1)
    weights = pseudo_weights(N=25, sort=True)
    bam = BinaryAcceptance(pseudo, weights=weights)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False)
    whdfs.find_paths(verbose=False)
    hdfs = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False)
    hdfs.find_paths(verbose=False)
    assert whdfs.res == hdfs.res


def test_basic_matrix():
    pseudo = np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]], dtype=bool)
    weights = None
    bam = BinaryAcceptance(pseudo, weights=weights)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    whdfs.find_paths(verbose=False)
    hdfs = HDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    hdfs.find_paths(verbose=False)
    assert whdfs.best.path == {0, 1, 2}
    assert whdfs.best.weight == 3
    assert whdfs.res == hdfs.res


def test_basic_matrix2():
    corr_matrix = np.array([[1.0, 0.005, 0.005],
                            [0.005, 1.0, 0.005],
                            [0.005, 0.005, 1.0]])
    weights = None
    threshold = 0.01
    bam = BinaryAcceptance(corr_matrix, weights=weights, threshold=threshold)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    whdfs.find_paths(verbose=False)
    hdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    hdfs.find_paths(verbose=False)
    assert whdfs.best.path == {0, 1, 2}
    assert whdfs.best.weight == 3
    assert whdfs.res == hdfs.res
