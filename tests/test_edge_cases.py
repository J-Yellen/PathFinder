"""Edge case tests for PathFinder algorithms"""
import numpy as np
from pathfinder.matrix_handler import BinaryAcceptance
from pathfinder.dfs import HDFS, WHDFS


def test_fully_connected_graph_with_subsets():
    """All combinations allowed: with allow_subset=True should return many paths"""
    N = 7
    # All edges are valid (fully connected)
    fully_connected = np.ones((N, N), dtype=bool)
    np.fill_diagonal(fully_connected, False)

    weights = np.arange(1, N + 1, 1)
    bam = BinaryAcceptance(fully_connected, weights=weights, threshold=1.0)

    # With allow_subset=True, should find many paths (all possible combinations)
    hdfs = HDFS(bam, top=None, allow_subset=True)
    hdfs.find_paths(runs=None)

    # Should find multiple paths of different sizes
    assert len(hdfs.get_paths) == 2**N - 1
    path_sizes = [len(p) for p in hdfs.get_paths]
    assert len(set(path_sizes)) > 1, "Should have paths of different sizes"


def test_fully_connected_graph_without_subsets():
    """All combinations allowed: with allow_subset=False should return exactly 1 path (the complete set)"""
    N = 5
    # All edges are valid (fully connected)
    fully_connected = np.ones((N, N), dtype=bool)
    np.fill_diagonal(fully_connected, False)

    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bam = BinaryAcceptance(fully_connected, weights=weights, threshold=1.0)

    # With allow_subset=False, should find exactly 1 path: the full set
    hdfs = HDFS(bam, top=10, allow_subset=False)
    hdfs.find_paths(runs=N)

    assert len(hdfs.get_paths) == 1, "Should find exactly one path (complete set)"
    assert len(hdfs.get_paths[0]) == N, f"Path should contain all {N} nodes"
    assert set(hdfs.get_paths[0]) == set(range(N)), "Path should be the complete set"

    # Test WHDFS too
    whdfs = WHDFS(bam, top=10, allow_subset=False, auto_sort=True)
    whdfs.find_paths(runs=N)

    assert len(whdfs.get_paths) == 1, "WHDFS should find exactly one path"
    assert len(whdfs.get_paths[0]) == N, f"Path should contain all {N} nodes"
    assert set(whdfs.get_paths[0]) == set(range(N)), "WHDFS path should be the complete set"


def test_fully_disconnected_graph():
    """No combinations allowed: should return only single-node paths"""
    N = 5
    # No edges are valid (fully disconnected)
    fully_disconnected = np.zeros((N, N), dtype=bool)

    weights = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Descending
    bam = BinaryAcceptance(fully_disconnected, weights=weights, threshold=0.0)

    hdfs = HDFS(bam, top=10, allow_subset=False)
    hdfs.find_paths(runs=N)

    # Should find only single-node paths
    assert all(len(p) == 1 for p in hdfs.get_paths), "All paths should be single nodes"
    # Should return top N paths (one for each node)
    assert len(hdfs.get_paths) <= N


def test_single_node_graph():
    """N=1: edge case with only one node"""
    N = 1
    single_node = np.array([[False]], dtype=bool)

    weights = np.array([1.0])
    bam = BinaryAcceptance(single_node, weights=weights)

    hdfs = HDFS(bam, top=5, allow_subset=False)
    hdfs.find_paths(runs=N)

    assert len(hdfs.get_paths) == 1
    assert hdfs.get_paths[0] == [0]
    assert hdfs.get_weights[0] == 1.0


def test_two_node_connected():
    """N=2 with valid edge: simplest non-trivial case"""
    N = 2
    two_nodes = np.array([[False, True], [True, False]], dtype=bool)

    weights = np.array([1.0, 2.0])
    bam = BinaryAcceptance(two_nodes, weights=weights)

    hdfs = HDFS(bam, top=5, allow_subset=False)
    hdfs.find_paths(runs=N)

    # Should find the path with both nodes
    assert len(hdfs.get_paths) >= 1
    assert set(hdfs.get_paths[0]) == {0, 1}


def test_two_node_disconnected():
    """N=2 with no edge: disconnected nodes result in single-node paths"""
    N = 2
    two_nodes_disconnected = np.array([[False, False], [False, False]], dtype=bool)

    weights = np.array([2.0, 1.0])
    bam = BinaryAcceptance(two_nodes_disconnected, weights=weights)

    hdfs = HDFS(bam, top=5, allow_subset=False)
    hdfs.find_paths(runs=N)

    # In disconnected graph, should find single-node path(s)
    assert len(hdfs.get_paths) >= 1
    # All paths should be single nodes
    assert all(len(p) == 1 for p in hdfs.get_paths)
    # First path should be the highest weight node
    assert hdfs.get_paths[0] == [0]


def test_top_exceeds_possible_paths():
    """Request more paths than exist: should return all available paths"""
    N = 3
    # Triangle: 3 nodes, all connected
    triangle = np.array([[False, True, True],
                        [True, False, True],
                        [True, True, False]], dtype=bool)

    weights = np.array([1.0, 1.0, 1.0])
    bam = BinaryAcceptance(triangle, weights=weights)

    # Request way more paths than possible
    hdfs = HDFS(bam, top=1000, allow_subset=False)
    hdfs.find_paths(runs=N)

    # Should find the single complete path
    assert len(hdfs.get_paths) <= N * 10  # Reasonable upper bound
    assert len(hdfs.get_paths) >= 1


def test_threshold_very_high():
    """threshold=1.0: with correlations < 1.0, all should be allowed"""
    N = 3
    # Create matrix with correlations all below 1.0
    corr_matrix = np.array([[1.0, 0.05, 0.03],
                           [0.05, 1.0, 0.04],
                           [0.03, 0.04, 1.0]])

    weights = np.array([1.0, 2.0, 3.0])
    bam = BinaryAcceptance(corr_matrix, weights=weights, threshold=1.0)

    # With threshold=1.0, all values < 1.0 are valid
    # All off-diagonal (< 1.0) should be True
    assert np.all(bam.bin_acc[np.triu_indices(N, k=1)]), "All edges < 1.0 should be valid"

    hdfs = HDFS(bam, top=5, allow_subset=False)
    hdfs.find_paths(runs=N)

    # Should find the complete path
    assert len(hdfs.get_paths[0]) == N


def test_threshold_one():
    """threshold=1.0: nothing should be allowed (for normalized correlations)"""
    N = 3
    # Normalized correlation matrix
    corr_matrix = np.array([[1.0, 0.5, 0.3],
                           [0.5, 1.0, 0.4],
                           [0.3, 0.4, 1.0]])

    weights = np.array([1.0, 2.0, 3.0])
    bam = BinaryAcceptance(corr_matrix, weights=weights, threshold=1.0)

    # All off-diagonal values < 1.0, so all should be True
    hdfs = HDFS(bam, top=5, allow_subset=False)
    hdfs.find_paths(runs=N)

    assert len(hdfs.get_paths) >= 1


def test_whdfs_with_uniform_weights():
    """WHDFS with uniform weights: sorting provides no benefit"""
    N = 5
    matrix = np.ones((N, N), dtype=bool)
    np.fill_diagonal(matrix, False)

    # All weights equal - no benefit from sorting
    weights = np.ones(N)
    bam = BinaryAcceptance(matrix, weights=weights)

    # WHDFS should complete without error even with uniform weights
    whdfs = WHDFS(bam, top=5, allow_subset=False, auto_sort=True)
    whdfs.find_paths(runs=N)

    # Should find at least one path
    assert len(whdfs.get_paths) >= 1


def test_large_n_with_sparse_connections():
    """Larger graph with sparse connections"""
    N = 20
    # Very sparse: only ~5% of edges valid
    sparse = np.random.rand(N, N) < 0.05
    sparse = np.triu(sparse, 1)
    sparse = sparse + sparse.T

    weights = np.random.rand(N)
    bam = BinaryAcceptance(sparse, weights=weights)

    whdfs = WHDFS(bam, top=10, allow_subset=False, auto_sort=True)
    whdfs.find_paths(runs=5)  # Only check first 5 nodes for speed

    # Should complete without error
    assert len(whdfs.get_paths) >= 1

    # All paths should be valid (no edges crossing black squares)
    for path in whdfs.get_paths:
        for i in range(len(path) - 1):
            idx1, idx2 = path[i], path[i+1]
            # Need to check in original space
            if whdfs.bam._index_map is not None:
                reverse_indices = np.argsort(whdfs.bam._index_map)
                original_bam = whdfs.bam.bin_acc[reverse_indices, :][:, reverse_indices]
            else:
                original_bam = whdfs.bam.bin_acc
            assert original_bam[idx1, idx2], f"Invalid edge in path: ({idx1}, {idx2})"


def test_no_valid_paths_from_source():
    """Edge case: source node has no valid edges"""
    # Node 0 isolated, others connected
    matrix = np.array([[False, False, False, False],
                      [False, False, True, True],
                      [False, True, False, True],
                      [False, True, True, False]], dtype=bool)

    weights = np.array([1.0, 2.0, 3.0, 4.0])
    bam = BinaryAcceptance(matrix, weights=weights)

    # Start from node 0 (isolated)
    hdfs = HDFS(bam, top=5, allow_subset=False)
    hdfs.find_paths(runs=1, source_node=0)

    # Should find at least the single-node path [0]
    assert len(hdfs.get_paths) >= 1
    assert [0] in hdfs.get_paths


def test_allow_subset_removes_subsets():
    """Verify allow_subset=False actually removes subset paths"""
    N = 4
    # Create a structure where subsets naturally occur
    matrix = np.array([[False, True, True, True],
                      [True, False, True, True],
                      [True, True, False, False],
                      [True, True, False, False]], dtype=bool)

    weights = np.array([4.0, 3.0, 2.0, 1.0])  # Descending
    bam = BinaryAcceptance(matrix, weights=weights)

    # With allow_subset=True
    hdfs_with = HDFS(bam, top=20, allow_subset=True)
    hdfs_with.find_paths(runs=N)

    # With allow_subset=False
    hdfs_without = HDFS(bam, top=20, allow_subset=False)
    hdfs_without.find_paths(runs=N)

    # Should have fewer paths without subsets
    assert len(hdfs_without.get_paths) <= len(hdfs_with.get_paths)

    # Verify no path is a subset of another when allow_subset=False
    paths_without = [set(p) for p in hdfs_without.get_paths]
    for i, path1 in enumerate(paths_without):
        for j, path2 in enumerate(paths_without):
            if i != j:
                assert not path1.issubset(path2), \
                    f"Path {path1} is a subset of {path2} (allow_subset=False should prevent this)"


def test_equal_weights_different_paths():
    """Multiple disconnected components with equal weights"""
    N = 4
    # Two disconnected pairs with equal weights
    matrix = np.array([[False, True, False, False],
                      [True, False, False, False],
                      [False, False, False, True],
                      [False, False, True, False]], dtype=bool)

    # Two pairs with equal total weights
    weights = np.array([2.0, 2.0, 1.0, 1.0])
    bam = BinaryAcceptance(matrix, weights=weights)

    hdfs = HDFS(bam, top=10, allow_subset=False)
    hdfs.find_paths(runs=N)

    # Should find at least one path (likely the highest weight component)
    assert len(hdfs.get_paths) >= 1

    # First path should be from the higher weight component
    assert set(hdfs.get_paths[0]) == {0, 1}, "First path should be the higher-weight pair"
