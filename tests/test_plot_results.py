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
    result = Results.from_dict({0: {'path': [0, 1, 2], 'weights': 2}})
    x, y = plot_results.format_path(result.get_paths[0])
    assert len(x) == 4 & len(y) == 4
    assert x == [[0, 0], [0, 1], [1, 1], [1, 2]]
    assert y == [[0, 1], [1, 1], [1, 2], [2, 2]]


def test_make_path():
    result = Results.from_dict({0: {'path': [0, 1, 2], 'weights': 2}})
    figure_data = plot_results.make_path([result.get_paths[0]])[0]
    assert isinstance(figure_data, dict)
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
    fig, axis = plot_results.plot(bam, result, size=12)
    assert isinstance(fig, figure.Figure)
    assert isinstance(axis, axes.Axes)
