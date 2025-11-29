import numpy as np
from matplotlib import figure, axes
from pathfinder.matrix_handler import BinaryAcceptance
from pathfinder import plot_results
from pathfinder import Results

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
