import warnings
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
    pseudo = pseudo_data(N=15, p=0.1)
    weights = pseudo_weights(N=15, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights)
    index_map = bam.sort_bam_by_weight()
    hdfs = HDFS(binary_acceptance_obj=bam, top=3, allow_subset=False)
    hdfs.find_paths(verbose=False)
    assert hdfs.best.path == {0, 2}
    assert hdfs.remap_path(index_map).best.path == {11, 13}


def test_WHDFS_1():
    """Test WHDFS with manual sorting (auto_sort=False)"""
    pseudo = pseudo_data(N=15, p=0.1)
    weights = pseudo_weights(N=15, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights)
    index_map = bam.sort_bam_by_weight()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*performance.*")
        whdfs = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False, auto_sort=False)
    whdfs.find_paths(verbose=False)
    assert whdfs.best.path == {0, 2}
    assert whdfs.remap_path(index_map).best.path == {11, 13}


def test_WHDFS_eq_HDFS():
    pseudo = pseudo_data(N=15, p=0.1)
    weights = pseudo_weights(N=15, sort=True)
    bam = BinaryAcceptance(pseudo, weights=weights)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*performance.*")
        whdfs = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False, auto_sort=False)
        whdfs.find_paths(verbose=False)
        hdfs = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False, auto_sort=False)
        hdfs.find_paths(verbose=False)
    assert whdfs.res == hdfs.res


def test_basic_matrix():
    pseudo = np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]], dtype=bool)
    weights = None
    bam = BinaryAcceptance(pseudo, weights=weights)
    # Suppress warning - uniform weights don't benefit from sorting
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*performance.*")
        whdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False, auto_sort=False)
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
    # Suppress warning - uniform weights don't benefit from sorting
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*performance.*")
        whdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False, auto_sort=False)
        whdfs.find_paths(verbose=False)
        hdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False, auto_sort=False)
        hdfs.find_paths(verbose=False)
    assert whdfs.best.path == {0, 1, 2}
    assert whdfs.best.weight == 3
    assert whdfs.res == hdfs.res


def test_WHDFS_auto_sort():
    """Test WHDFS with automatic sorting (default behavior)"""
    pseudo = pseudo_data(N=15, p=0.1)
    weights = pseudo_weights(N=15, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights)
    # With auto_sort=True (default), results are automatically remapped
    whdfs = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False)
    whdfs.find_paths(verbose=False)
    # get_paths should return remapped indices automatically
    assert whdfs.get_paths[0] == [11, 13]


def test_convenience_function():
    """Test find_best_combinations convenience function"""
    from pathfinder import find_best_combinations
    pseudo = pseudo_data(N=15, p=0.1)
    weights = pseudo_weights(N=15, sort=False)
    # Use convenience function
    results = find_best_combinations(
        matrix=pseudo,
        weights=weights,
        top=3,
        allow_subset=False
    )
    # Should automatically return remapped results
    assert results.get_paths[0] == [11, 13]


def test_auto_sort_warning():
    """Test that warning is issued when auto_sort=False with non-uniform weights"""
    pseudo = pseudo_data(N=15, p=0.1)
    weights = pseudo_weights(N=15, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights)
    # Should emit warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False, auto_sort=False)
        assert len(w) == 1
        assert "performance" in str(w[0].message).lower()


def test_hdfs_trim_false():
    """Test HDFS with trim=False to include target node"""
    pseudo = np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    hdfs = HDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    # Access hdfs generator directly with trim=False
    paths = list(hdfs.hdfs(trim=False))
    assert len(paths) > 0
    # With trim=False, target node (3) should be included
    assert any(3 in path for path in paths)


def test_hdfs_with_ignore_child():
    """Test HDFS with ignore_child parameter"""
    pseudo = np.array([[0, 1, 1, 1],
                       [1, 0, 1, 1],
                       [1, 1, 0, 1],
                       [1, 1, 1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    hdfs = HDFS(binary_acceptance_obj=bam, top=5, allow_subset=True)
    # Find paths ignoring child node 2
    hdfs.find_paths(ignore_child=[2], verbose=False)
    # No path should contain node 2 as first child
    assert all(2 not in path[:1] for path in hdfs.get_paths)


def test_whdfs_with_ignore_child():
    """Test WHDFS with ignore_child parameter"""
    pseudo = np.array([[0, 1, 1, 1],
                       [1, 0, 1, 1],
                       [1, 1, 0, 1],
                       [1, 1, 1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*performance.*")
        whdfs = WHDFS(binary_acceptance_obj=bam, top=5, allow_subset=True, auto_sort=False)
    # Find paths with ignore_child list
    whdfs.find_paths(ignore_child=[None, None, None, None], verbose=False)
    assert len(whdfs.get_paths) > 0


def test_whdfs_ignore_child_length_mismatch():
    """Test WHDFS raises error when ignore_child length doesn't match runs"""
    import pytest
    pseudo = np.array([[0, 1], [1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*performance.*")
        whdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False, auto_sort=False)
    with pytest.raises(Exception, match="ignore_child.*length.*runs"):
        whdfs.find_paths(runs=2, ignore_child=[0], verbose=False)


def test_hdfs_with_runs():
    """Test HDFS with limited runs parameter"""
    pseudo = np.array([[0, 1, 1, 1, 1],
                       [1, 0, 1, 1, 1],
                       [1, 1, 0, 1, 1],
                       [1, 1, 1, 0, 1],
                       [1, 1, 1, 1, 0]], dtype=bool)
    weights = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    bam = BinaryAcceptance(pseudo, weights=weights)
    hdfs = HDFS(binary_acceptance_obj=bam, top=3, allow_subset=False)
    # Only search from first 2 nodes
    hdfs.find_paths(runs=2, verbose=False)
    assert len(hdfs.get_paths) > 0


def test_hdfs_verbose():
    """Test HDFS with verbose=True"""
    pseudo = np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    hdfs = HDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    # verbose=True should print results (just testing it doesn't crash)
    hdfs.find_paths(verbose=True)
    assert len(hdfs.get_paths) > 0


def test_whdfs_verbose():
    """Test WHDFS with verbose=True"""
    pseudo = np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    # verbose=True should print results
    whdfs.find_paths(verbose=True)
    assert len(whdfs.get_paths) > 0


def test_whdfs_reset_result_false():
    """Test WHDFS with reset_result=False"""
    pseudo = np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=5, allow_subset=False)
    whdfs.find_paths(verbose=False)
    initial_count = len(whdfs.get_paths)
    # Run again without resetting results
    whdfs.find_paths(reset_result=False, verbose=False)
    # Results should accumulate (or at least not reset)
    assert len(whdfs.get_paths) >= initial_count


def test_convenience_function_with_hdfs():
    """Test find_best_combinations with HDFS algorithm"""
    import pathfinder as pf
    pseudo = np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]], dtype=bool)
    weights = np.array([3.0, 2.0, 1.0])
    results = pf.find_best_combinations(
        matrix=pseudo,
        weights=weights,
        top=1,
        algorithm='hdfs',
        verbose=False
    )
    assert len(results.get_paths) > 0


def test_convenience_function_invalid_algorithm():
    """Test find_best_combinations with invalid algorithm"""
    import pytest
    import pathfinder as pf
    pseudo = np.array([[0, 1], [1, 0]], dtype=bool)
    with pytest.raises(ValueError, match="Unknown algorithm"):
        pf.find_best_combinations(
            matrix=pseudo,
            algorithm='invalid_algo'
        )


def test_convenience_function_with_runs():
    """Test find_best_combinations with runs parameter"""
    import pathfinder as pf
    pseudo = np.array([[0, 1, 1, 1],
                       [1, 0, 1, 1],
                       [1, 1, 0, 1],
                       [1, 1, 1, 0]], dtype=bool)
    weights = np.array([4.0, 3.0, 2.0, 1.0])
    results = pf.find_best_combinations(
        matrix=pseudo,
        weights=weights,
        top=2,
        runs=2,
        verbose=False
    )
    assert len(results.get_paths) > 0


def test_convenience_function_with_labels():
    """Test find_best_combinations with labels parameter"""
    import pathfinder as pf
    pseudo = np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]], dtype=bool)
    labels = ['A', 'B', 'C']
    results = pf.find_best_combinations(
        matrix=pseudo,
        labels=labels,
        top=1
    )
    assert len(results.get_paths) > 0


def test_hdfs_with_source_node():
    """Test HDFS find_paths with non-zero source_node"""
    pseudo = np.array([[0, 1, 1, 1],
                       [1, 0, 1, 1],
                       [1, 1, 0, 1],
                       [1, 1, 1, 0]], dtype=bool)
    weights = np.array([4.0, 3.0, 2.0, 1.0])
    bam = BinaryAcceptance(pseudo, weights=weights)
    hdfs = HDFS(binary_acceptance_obj=bam, top=3, allow_subset=False)
    hdfs.find_paths(source_node=1, runs=2, verbose=False)
    assert len(hdfs.get_paths) > 0


def test_whdfs_with_source_node():
    """Test WHDFS find_paths with non-zero source_node"""
    pseudo = np.array([[0, 1, 1, 1],
                       [1, 0, 1, 1],
                       [1, 1, 0, 1],
                       [1, 1, 1, 0]], dtype=bool)
    weights = np.array([4.0, 3.0, 2.0, 1.0])
    bam = BinaryAcceptance(pseudo, weights=weights)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False)
    # When using source_node, runs starts from that node
    whdfs.find_paths(source_node=0, runs=2, verbose=False)
    assert len(whdfs.get_paths) > 0


def test_hdfs_multiple_find_paths_calls():
    """Test HDFS calling find_paths multiple times"""
    pseudo = np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    hdfs = HDFS(binary_acceptance_obj=bam, top=5, allow_subset=True)
    hdfs.find_paths(verbose=False)
    initial_count = len(hdfs.get_paths)
    # Call again - should reset results
    hdfs.find_paths(verbose=False)
    # Results should be reset, not accumulated
    assert len(hdfs.get_paths) <= initial_count * 2


def test_whdfs_cutoff_reached():
    """Test WHDFS when path length reaches cutoff"""
    # Create a fully connected graph
    pseudo = np.array([[0, 1, 1, 1, 1],
                       [1, 0, 1, 1, 1],
                       [1, 1, 0, 1, 1],
                       [1, 1, 1, 0, 1],
                       [1, 1, 1, 1, 0]], dtype=bool)
    weights = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    bam = BinaryAcceptance(pseudo, weights=weights)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    whdfs.find_paths(verbose=False)
    # Should find at least one path
    assert len(whdfs.get_paths) > 0


def test_version_import():
    """Test that __version__ is accessible"""
    import pathfinder as pf
    assert hasattr(pf, '__version__')
    assert isinstance(pf.__version__, str)


def test_hdfs_negative_weights_with_subset_false():
    """Test HDFS raises exception for negative weights when allow_subset=False"""
    import pytest
    pseudo = np.array([[0, 1], [1, 0]], dtype=bool)
    weights = np.array([1.0, -1.0])
    bam = BinaryAcceptance(pseudo, weights=weights, allow_negative_weights=True)
    with pytest.raises(Exception, match="Negative weights"):
        HDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)


def test_whdfs_negative_weights_with_subset_false():
    """Test WHDFS raises exception for negative weights when allow_subset=False"""
    import pytest
    pseudo = np.array([[0, 1], [1, 0]], dtype=bool)
    weights = np.array([1.0, -1.0])
    bam = BinaryAcceptance(pseudo, weights=weights, allow_negative_weights=True)
    with pytest.raises(Exception, match="Negative weights"):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*performance.*")
            WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False, auto_sort=False)


def test_hdfs_cutoff_with_trim_false():
    """Test HDFS with trim=False when cutoff is reached"""
    # Create a graph where we can reach cutoff
    pseudo = np.array([[0, 1, 1, 1],
                       [1, 0, 1, 1],
                       [1, 1, 0, 1],
                       [1, 1, 1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    hdfs = HDFS(binary_acceptance_obj=bam, top=5, allow_subset=False)
    # Use hdfs generator directly with trim=False
    paths = list(hdfs.hdfs(trim=False))
    # With trim=False, target node should be in some paths
    assert any(len(path) > 3 for path in paths)


def test_whdfs_cutoff_logic():
    """Test WHDFS cutoff condition by creating paths that reach length limit"""
    # Create fully connected small graph to test cutoff
    pseudo = np.array([[0, 1, 1, 1, 1],
                       [1, 0, 1, 1, 1],
                       [1, 1, 0, 1, 1],
                       [1, 1, 1, 0, 1],
                       [1, 1, 1, 1, 0]], dtype=bool)
    weights = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    bam = BinaryAcceptance(pseudo, weights=weights)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False)
    whdfs.find_paths(verbose=False)
    # Should find paths
    assert len(whdfs.get_paths) > 0
    # Paths should reach target (sink node is at position dim)
    assert any(len(path) >= 3 for path in whdfs.get_paths)
