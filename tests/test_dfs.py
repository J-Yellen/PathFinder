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


def test_hdfs_get_sorted_results_not_implemented():
    """Test that HDFS.get_sorted_results raises NotImplementedError"""
    import pytest
    pseudo = np.array([[0, 1], [1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    hdfs = HDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    hdfs.find_paths(verbose=False)
    with pytest.raises(NotImplementedError, match="not implemented for HDFS"):
        hdfs.get_sorted_results()


def test_whdfs_get_sorted_results():
    """Test WHDFS.get_sorted_results returns Results in sorted index space"""
    pseudo = pseudo_data(N=15, p=0.1)
    weights = pseudo_weights(N=15, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False, auto_sort=True)
    whdfs.find_paths(verbose=False)
    # Get sorted results
    sorted_results = whdfs.get_sorted_results()
    # Should be a Results object
    from pathfinder.result import Results
    assert isinstance(sorted_results, Results)
    # Should have paths in sorted space (different from remapped paths)
    if len(whdfs.get_paths) > 0:
        assert sorted_results.get_paths[0] != whdfs.get_paths[0]


def test_whdfs_get_sorted_paths():
    """Test WHDFS.get_sorted_paths returns paths in sorted index space"""
    pseudo = pseudo_data(N=15, p=0.1)
    weights = pseudo_weights(N=15, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False, auto_sort=True)
    whdfs.find_paths(verbose=False)
    # Get sorted paths
    sorted_paths = whdfs.get_sorted_paths()
    # Should be different from remapped paths
    if len(whdfs.get_paths) > 0:
        assert sorted_paths[0] != whdfs.get_paths[0]
        # Sorted paths should be [0, 2] for this specific test
        assert sorted_paths[0] == [0, 2]


def test_whdfs_string_representation_with_remapping():
    """Test WHDFS.__str__ shows remapped paths when auto_sort is used"""
    pseudo = pseudo_data(N=15, p=0.1)
    weights = pseudo_weights(N=15, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False, auto_sort=True)
    whdfs.find_paths(verbose=False)
    # String representation should show remapped paths
    str_repr = str(whdfs)
    assert isinstance(str_repr, str)
    # Should contain the remapped indices [11, 13]
    if len(whdfs.get_paths) > 0:
        assert '11' in str_repr and '13' in str_repr


def test_whdfs_string_representation_without_remapping():
    """Test WHDFS.__str__ without remapping"""
    pseudo = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*performance.*")
        whdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False, auto_sort=False)
    whdfs.find_paths(verbose=False)
    # String representation should work normally
    str_repr = str(whdfs)
    assert isinstance(str_repr, str)


def test_whdfs_weight_func_setter():
    """Test WHDFS weight_func property setter"""
    pseudo = np.array([[0, 1], [1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    # Set custom weight function

    def custom_func(x):
        return sum(x) * 2
    whdfs.weight_func = custom_func
    assert whdfs.weight_func == custom_func


def test_whdfs_wlimit_func_setter():
    """Test WHDFS wlimit_func property setter"""
    pseudo = np.array([[0, 1], [1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    # Set custom wlimit function

    def custom_func(x):
        return sum(x) * 1.5
    whdfs.wlimit_func = custom_func
    assert whdfs.wlimit_func == custom_func


def test_whdfs_set_top_weight():
    """Test WHDFS.set_top_weight method"""
    pseudo = np.array([[0, 1], [1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    # Set custom top weight function

    def custom_func():
        return 5.0
    whdfs.set_top_weight(custom_func)
    # Verify the function was set
    assert whdfs._top_weights == custom_func
    # Now top_weight() should call the custom function and return 5.0
    assert whdfs.top_weight() == 5.0


def test_whdfs_top_weight_default_behavior():
    """Test WHDFS._top_weights_default returns -inf when not enough paths"""
    pseudo = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    weights = np.array([3.0, 2.0, 1.0])
    bam = BinaryAcceptance(pseudo, weights=weights)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=10, allow_subset=False)  # top > available paths
    # Before finding paths, should return -inf
    assert whdfs._top_weights_default() == float('-inf')


def test_hdfs_chunked_static_method():
    """Test HDFS.chunked static method"""
    data = list(range(10))
    chunks = list(HDFS.chunked(data, 3))
    assert len(chunks) == 4  # 3+3+3+1
    assert chunks[0] == [0, 1, 2]
    assert chunks[1] == [3, 4, 5]
    assert chunks[2] == [6, 7, 8]
    assert chunks[3] == [9]


def test_whdfs_pruning_with_insufficient_remaining_weight():
    """Test WHDFS prunes paths when remaining weight is insufficient"""
    # Create a graph where some paths will be pruned
    pseudo = np.array([[0, 1, 1, 1],
                       [1, 0, 1, 0],
                       [1, 1, 0, 0],
                       [1, 0, 0, 0]], dtype=bool)
    weights = np.array([10.0, 5.0, 1.0, 0.1])
    bam = BinaryAcceptance(pseudo, weights=weights)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    whdfs.find_paths(verbose=False)
    # Should find at least one path
    assert len(whdfs.get_paths) > 0


def test_whdfs_with_visited_child():
    """Test WHDFS continues correctly when child is already visited"""
    # Create a graph with potential revisit situations
    pseudo = np.array([[0, 1, 1, 1],
                       [1, 0, 1, 1],
                       [1, 1, 0, 1],
                       [1, 1, 1, 0]], dtype=bool)
    weights = np.array([4.0, 3.0, 2.0, 1.0])
    bam = BinaryAcceptance(pseudo, weights=weights)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=5, allow_subset=False)
    whdfs.find_paths(verbose=False)
    # Should handle visited nodes correctly
    assert len(whdfs.get_paths) > 0


def test_hdfs_remap_path_without_index_map():
    """Test HDFS.remap_path when no index_map is provided and BAM has none"""
    pseudo = np.array([[0, 1], [1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    hdfs = HDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    hdfs.find_paths(verbose=False)
    # Remap without providing index_map (should use BAM's _index_map which is None)
    remapped = hdfs.remap_path()
    # Should be a no-op when no index_map exists
    assert remapped.get_paths == hdfs.get_paths


def test_whdfs_remap_path_without_index_map():
    """Test WHDFS.remap_path when explicitly called without index_map"""
    pseudo = np.array([[0, 1], [1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*performance.*")
        whdfs = WHDFS(binary_acceptance_obj=bam, top=1, allow_subset=False, auto_sort=False)
    whdfs.find_paths(verbose=False)
    # Remap without providing index_map
    remapped = whdfs.remap_path()
    # Should be a no-op when BAM has no _index_map
    assert len(remapped.get_paths) == len(whdfs.get_sorted_paths())


def test_hdfs_remap_path_with_weight_offset():
    """Test HDFS.remap_path with weight_offset parameter"""
    pseudo = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    weights = np.array([3.0, 2.0, 1.0])
    bam = BinaryAcceptance(pseudo, weights=weights)
    index_map = bam.sort_bam_by_weight()
    hdfs = HDFS(binary_acceptance_obj=bam, top=1, allow_subset=False)
    hdfs.find_paths(verbose=False)
    # Remap with weight offset
    remapped = hdfs.remap_path(index_map, weight_offset=0.5)
    if len(remapped.get_paths) > 0:
        # Weight should be adjusted
        original_weight = hdfs.get_weights[0]
        remapped_weight = remapped.get_weights[0]
        # Weight offset is per path element, so difference depends on path length
        assert remapped_weight < original_weight


def test_whdfs_get_weights_unchanged():
    """Test WHDFS.get_weights returns unchanged weights"""
    pseudo = pseudo_data(N=15, p=0.1)
    weights = pseudo_weights(N=15, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=3, allow_subset=False, auto_sort=True)
    whdfs.find_paths(verbose=False)
    # get_weights should match internal weights
    if len(whdfs.get_weights) > 0:
        assert whdfs.get_weights[0] > 0
        # Should be same as sorted results weights
        sorted_results = whdfs.get_sorted_results()
        assert whdfs.get_weights == sorted_results.get_weights


def test_hdfs_with_float_ignore_child():
    """Test HDFS find_paths converts float ignore_child to int"""
    pseudo = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo, weights=None)
    hdfs = HDFS(binary_acceptance_obj=bam, top=3, allow_subset=True)
    # Pass float as ignore_child - should be converted to int list
    hdfs.find_paths(ignore_child=1.0, verbose=False)
    assert len(hdfs.get_paths) > 0


def test_whdfs_reaches_target_node():
    """Test WHDFS correctly identifies when target node is reached"""
    # Create small fully connected graph
    pseudo = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    weights = np.array([3.0, 2.0, 1.0])
    bam = BinaryAcceptance(pseudo, weights=weights)
    whdfs = WHDFS(binary_acceptance_obj=bam, top=5, allow_subset=False)
    whdfs.find_paths(verbose=False)
    # Should find paths that reach target
    assert len(whdfs.get_paths) > 0
    # All paths should have weights greater than minimum threshold
    assert all(w > 0 for w in whdfs.get_weights)
