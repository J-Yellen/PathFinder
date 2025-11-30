import numpy as np
from matplotlib import figure, axes
from pathfinder.matrix_handler import BinaryAcceptance
from pathfinder import plot_results
from pathfinder import Results, WHDFS

np.random.seed(1)


def pseudo_data(N=25, p=0.05) -> np.ndarray:
    pseudo = np.triu(np.random.choice([True, False], size=(N, N), p=[p, 1 - p]), 1)
    pseudo += pseudo.T
    return pseudo


def pseudo_weights(N=25, sort=True) -> np.ndarray:
    if sort:
        return np.sort(np.random.rand(N))[::-1]
    else:
        return np.random.rand(N)


def test_format_path():
    result = Results.from_dict({0: {'path': [0, 1, 2], 'weight': 2}})
    formatted = plot_results.format_path(result.get_paths[0])
    x, y = formatted[0], formatted[1]
    assert len(x) == 4 and len(y) == 4
    expected_x = [[0, 0], [0, 1], [1, 1], [1, 2]]
    expected_y = [[0, 1], [1, 1], [1, 2], [2, 2]]
    assert np.array_equal(x, expected_x)
    assert np.array_equal(y, expected_y)


def test_make_path():
    result = Results.from_dict({0: {'path': [0, 1, 2], 'weight': 2}})
    figure_data = plot_results.make_path([result.get_paths[0]])[0]
    assert isinstance(figure_data, dict)
    assert 'x' in figure_data and 'y' in figure_data and 'color' in figure_data
    assert figure_data['x'] == [0, 0, 0, 1, 1, 1, 1, 2]
    assert figure_data['y'] == [0, 1, 1, 1, 1, 2, 2, 2]


def test_plot():
    result = Results.from_dict({0: {'path': [0, 1, 2], 'weight': 1.9},
                                1: {'path': [0, 1, 4], 'weight': 1.8}})
    p, N = 0.5, 5
    pseudo = pseudo_data(N, p)
    weights = pseudo_weights(N, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights)
    _ = bam.sort_bam_by_weight()
    result_plot = plot_results.plot(bam, result, size=12)
    assert result_plot is not None
    fig, axis = result_plot
    assert isinstance(fig, figure.Figure)
    assert isinstance(axis, axes.Axes)


def test_add_sink_data():
    """Test add_sink_data function with results"""
    p, N = 0.5, 5
    pseudo = pseudo_data(N, p)
    weights = pseudo_weights(N, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights)
    result = Results.from_dict({0: {'path': [0, 1, 2], 'weight': 1.9}})

    # Test with results and labels
    dat, new_result, labels = plot_results.add_sink_data(bam, result, xy_labels=['A', 'B', 'C', 'D', 'E'])
    assert dat.shape == (N + 1, N + 1)
    assert 'Sink' in labels
    assert new_result is not None

    # Test without labels (generates default labels including sink)
    dat2, new_result2, labels2 = plot_results.add_sink_data(bam, result)
    assert labels2[-1] == 'Sink'
    assert len(labels2) > N  # Includes sink node


def test_plot_with_custom_labels():
    """Test plot function with custom labels"""
    result = Results.from_dict({0: {'path': [0, 1], 'weight': 1.0}})
    pseudo = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo)
    labels = ['Feature A', 'Feature B', 'Feature C']

    fig, axis = plot_results.plot(bam, result, xy_labels=labels, size=10)
    assert fig is not None
    assert axis is not None


def test_plot_without_results():
    """Test plot function without results (just BAM)"""
    pseudo = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
    bam = BinaryAcceptance(pseudo)

    fig, axis = plot_results.plot(bam, results=None, size=8)
    assert fig is not None
    assert axis is not None


def test_plot_sorted_parameter():
    """Test plot_sorted parameter for comparing HDFS and WHDFS visualizations"""
    from pathfinder.dfs import HDFS, WHDFS

    # Create test data
    p, N = 0.1, 10
    pseudo = pseudo_data(N, p)
    weights = pseudo_weights(N, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights, threshold=0.05)

    # Run HDFS and WHDFS
    hdfs = HDFS(bam, top=5)
    hdfs.find_paths()

    whdfs = WHDFS(bam, top=5, auto_sort=True)
    whdfs.find_paths()

    # Test default behavior (plot_sorted=False) - both in original space
    fig1, ax1 = plot_results.plot(bam, hdfs, plot_sorted=False, size=10)
    fig2, ax2 = plot_results.plot(bam, whdfs, plot_sorted=False, size=10)
    assert fig1 is not None and fig2 is not None

    # Verify WHDFS returns paths in original index space (auto-remapped)
    if len(whdfs.get_paths) > 0 and whdfs.bam._index_map is not None:
        # get_paths should be in original space
        original_paths = whdfs.get_paths
        # get_sorted_paths should be in sorted space
        sorted_paths = whdfs.get_sorted_paths()
        # They should be the same length but potentially different indices
        assert len(original_paths) == len(sorted_paths)

    # Test sorted visualization (plot_sorted=True)
    fig3, ax3 = plot_results.plot(bam, hdfs, plot_sorted=True, size=10)
    fig4, ax4 = plot_results.plot(bam, whdfs, plot_sorted=True, size=10)
    assert fig3 is not None and fig4 is not None

    # Test verbose output uses remapped paths
    if len(whdfs.res) > 0 and whdfs.bam._index_map is not None:
        # String representation should show original indices
        whdfs_str = str(whdfs)
        assert whdfs_str is not None and len(whdfs_str) > 0


def test_paths_dont_land_on_black_squares():
    """Critical test: verify paths never land on black squares (invalid edges)"""
    from pathfinder.dfs import HDFS, WHDFS

    # Create test data with clear structure
    N = 10
    pseudo = pseudo_data(N, p=0.3)
    weights = pseudo_weights(N, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights, threshold=0.05)

    # Test HDFS
    hdfs = HDFS(bam, top=5, allow_subset=False)
    hdfs.find_paths()

    if len(hdfs.get_paths) > 0:
        # Verify each path only uses valid edges
        for path in hdfs.get_paths:
            for i in range(len(path) - 1):
                idx1, idx2 = path[i], path[i+1]
                # Check the BAM in original space (unsorted)
                # Since plot_sorted=False, we compare against original BAM
                if bam._index_map is not None:
                    # Unsort to get original BAM
                    reverse_indices = np.argsort(bam._index_map)
                    original_bam = bam.bin_acc[reverse_indices, :][:, reverse_indices]
                else:
                    original_bam = bam.bin_acc

                # The edge (idx1, idx2) must be acceptable (True in BAM)
                assert original_bam[idx1, idx2], \
                    f"Path contains invalid edge ({idx1}, {idx2}) in original space"

    # Test WHDFS with auto_sort
    whdfs = WHDFS(bam, top=5, allow_subset=False, auto_sort=True)
    whdfs.find_paths()

    if len(whdfs.get_paths) > 0:
        # get_paths returns paths in ORIGINAL indices
        # Verify each path only uses valid edges in ORIGINAL BAM
        for path in whdfs.get_paths:
            for i in range(len(path) - 1):
                idx1, idx2 = path[i], path[i+1]
                # Unsort BAM to original space
                if whdfs.bam._index_map is not None:
                    reverse_indices = np.argsort(whdfs.bam._index_map)
                    original_bam = whdfs.bam.bin_acc[reverse_indices, :][:, reverse_indices]
                else:
                    original_bam = whdfs.bam.bin_acc

                # The edge must be valid in original BAM
                assert original_bam[idx1, idx2], \
                    f"WHDFS path contains invalid edge ({idx1}, {idx2}) in original space"


def test_highlight_top_path():
    """Test highlight_top_path parameter works correctly"""
    from pathfinder.dfs import HDFS, WHDFS

    # Create test data
    N = 8
    pseudo = pseudo_data(N, p=0.3)
    weights = pseudo_weights(N, sort=False)
    bam = BinaryAcceptance(pseudo, weights=weights, threshold=0.05)

    # Test with HDFS
    hdfs = HDFS(bam, top=3, allow_subset=False)
    hdfs.find_paths()

    if len(hdfs.get_paths) > 0:
        # Should work without error
        fig1, ax1 = plot_results.plot(bam, hdfs, highlight_top_path=True, size=8)
        assert fig1 is not None
        assert ax1 is not None

        # Without highlighting should also work
        fig2, ax2 = plot_results.plot(bam, hdfs, highlight_top_path=False, size=8)
        assert fig2 is not None
        assert ax2 is not None

    # Test with WHDFS and auto_sort
    whdfs = WHDFS(bam, top=3, allow_subset=False, auto_sort=True)
    whdfs.find_paths()

    if len(whdfs.get_paths) > 0:
        # Should work with highlighting in both plot modes
        fig3, ax3 = plot_results.plot(bam, whdfs, highlight_top_path=True,
                                      plot_sorted=False, size=8)
        assert fig3 is not None
        assert ax3 is not None

        fig4, ax4 = plot_results.plot(bam, whdfs, highlight_top_path=True,
                                      plot_sorted=True, size=8)
        assert fig4 is not None
        assert ax4 is not None

    # Test with labels
    labels = [f'Feature_{i}' for i in range(N)]
    fig5, ax5 = plot_results.plot(bam, hdfs, xy_labels=labels,
                                  highlight_top_path=True, size=8)
    assert fig5 is not None
    assert ax5 is not None


def test_axis_labels_parameter():
    """Test axis_labels parameter uses BAM labels correctly"""

    # Create test data with labels
    N = 6
    pseudo = pseudo_data(N, p=0.3)
    weights = pseudo_weights(N, sort=False)
    labels = [f'Feature_{chr(65+i)}' for i in range(N)]

    # Create BAM with labels
    bam = BinaryAcceptance(pseudo, weights=weights, labels=labels, threshold=0.05)

    whdfs = WHDFS(bam, top=3, allow_subset=False, auto_sort=True)
    whdfs.find_paths()

    # Test with axis_labels=True (should use BAM labels)
    if bam.labels is not None:
        fig1, ax1 = plot_results.plot(bam, whdfs, axis_labels=True, size=8)
        assert fig1 is not None
        assert ax1 is not None
        # Check that labels were applied
        assert len(ax1.get_xticklabels()) > 0

        # Test that xy_labels overrides axis_labels
        custom_labels = [f'Custom_{i}' for i in range(N)]
        fig2, ax2 = plot_results.plot(bam, whdfs, xy_labels=custom_labels,
                                      axis_labels=True, size=8)
        assert fig2 is not None
        assert ax2 is not None

        # Test axis_labels=False with no xy_labels (no labels shown)
        fig3, ax3 = plot_results.plot(bam, whdfs, axis_labels=False, size=8)
        assert fig3 is not None
        assert ax3 is not None


def test_plot_with_results_only():
    """Test plot() can extract BAM from WHDFS/HDFS objects"""
    # Create test data
    N = 6
    pseudo = pseudo_data(N, p=0.3)
    weights = pseudo_weights(N, sort=False)

    # Create WHDFS with BAM
    bam = BinaryAcceptance(pseudo, weights=weights, threshold=0.05)
    whdfs = WHDFS(bam, top=3, allow_subset=False, auto_sort=True)
    whdfs.find_paths()

    # Test simplified API - just pass results object
    fig1, ax1 = plot_results.plot(results=whdfs, size=8)
    assert fig1 is not None
    assert ax1 is not None

    # Test with optional parameters
    fig2, ax2 = plot_results.plot(results=whdfs, highlight_top_path=True, size=8)
    assert fig2 is not None
    assert ax2 is not None

    # Test that explicit bam still works (backward compatibility)
    fig3, ax3 = plot_results.plot(bam=bam, results=whdfs, size=8)
    assert fig3 is not None
    assert ax3 is not None


def test_plot_requires_bam():
    """Test plot() raises appropriate error when BAM cannot be extracted"""
    # Create Results object without bam attribute
    results = Results(top=5)

    # Should raise ValueError when bam is None and results has no bam
    import pytest
    with pytest.raises(ValueError, match="bam parameter is required"):
        plot_results.plot(results=results)

    # Should also raise when both are None
    with pytest.raises(ValueError, match="Either bam parameter or results with bam attribute"):
        plot_results.plot()
