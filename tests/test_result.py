import numpy as np
from pathfinder.result import Result, Results
from pathlib import Path

np.random.seed(1)


def sim_path(max_value: int = 10):
    length = np.random.randint(max_value, size=1)
    return set(np.random.randint(max_value, size=length))


def test_result_class():
    path = sim_path(max_value=10)
    weight = 4.0
    res1 = Result(path=path, weight=weight)
    res2 = Result(path=path, weight=weight - 1)
    assert max([res1, res2]).weight == weight
    assert min([res1, res2]).weight == weight - 1


def test_results_class():

    paths = [sim_path(max_value=20) for _ in range(100)]
    weights = [float(sum(p)) for p in paths]
    results = Results(paths, weights, top=10)
    assert results.best.weight == max(weights)
    assert results.best.path == paths[np.argmax(weights)]
    assert results.top == 10
    results.top = 20
    assert results.top == 20


def test_bisect_left():

    paths = [{i} for i in range(10)]
    weights = [float(i) for i in range(10)]
    results = Results(paths, weights, top=10)
    print(results._res)
    new_item1 = Result({6}, 5.5)
    new_item2 = Result({5}, 5)
    new_item3 = Result({10}, 10.0)
    idx1 = results._bisect_left(results._res, new_item1)
    idx2 = results._bisect_left(results._res, new_item2)
    idx3 = results._bisect_left(results._res, new_item3)
    assert idx1 == 4
    assert idx2 == 5
    assert idx3 == 0


def test_add_result():
    paths = [{i} for i in range(10) if i != 5]
    weights = [float(i) for i in range(10) if i != 5]
    results = Results(paths, weights, top=10)
    results.add_result({5}, 5.0, trim_to_top=True, bisect=True)
    results.add_result({10}, 10, trim_to_top=True, bisect=True)
    assert results.res[5] == Result({5}, 5.0)
    assert results.res[0] == Result({10}, 10)


def test_bulk_add_res():
    paths = [sim_path(max_value=20) for _ in range(100)]
    weights = [float(sum(p)) for p in paths]
    paths2 = [sim_path(max_value=20) for _ in range(100)] + [{0, 1000}]
    weights2 = [float(sum(p)) for p in paths] + [1000.0]
    results = Results(paths, weights, top=10)
    results.bulk_add_result(paths2, weights2)
    assert len(results.res) == 10
    assert results.best == Result({0, 1000}, 1000)


def test_remap_path():

    paths = [{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 5, 9}]
    results = Results(paths, [1, 1], top=2, allow_subset=True)
    map1 = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    map2 = dict(enumerate(map1))
    map_result1 = results.remap_path(map1)
    map_result2 = results.remap_path(map2)
    assert map_result1 == map_result2
    assert map_result1.get_paths[1] == sorted([map1[i] for i in paths[1]])
    assert map_result1.get_raw_paths[1] == {map1[i] for i in paths[1]}


def test_results_to_json_to_file():
    """Test Results JSON serialization to file"""
    import tempfile
    import os
    paths = [{0, 1, 2}, {1, 2}]
    weights = [3.0, 2.0]
    results = Results(paths, weights, top=2)
    # Serialize to JSON file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        results.to_json(Path(f.name))
        temp_path = f.name
    try:
        # Verify file exists and contains data
        assert os.path.exists(temp_path)
        assert os.path.getsize(temp_path) > 0
    finally:
        os.unlink(temp_path)


def test_results_equality():
    """Test Results equality comparison"""
    paths = [{0, 1}, {1, 2}]
    weights = [2.0, 2.0]
    results1 = Results(paths, weights, top=2)
    results2 = Results(paths, weights, top=2)
    assert results1 == results2


def test_results_not_equal_to_other_type():
    """Test Results not equal to non-Results object"""
    paths = [{0, 1}]
    weights = [1.0]
    results = Results(paths, weights, top=1)
    assert results != "not a Results object"
    assert results != 42


def test_results_empty_initialization():
    """Test Results with empty paths"""
    results = Results([], [], top=5)
    # Empty results still has one empty path
    assert len(results.get_paths) == 1
    assert results.get_paths[0] == []
    assert results.best is not None


def test_results_mismatched_lengths():
    """Test Results raises error when paths and weights lengths don't match"""
    import pytest
    with pytest.raises(ValueError, match="Unequal length"):
        Results([{0}], [1.0, 2.0], top=1)


def test_results_paths_without_weights():
    """Test Results.add_result with only path (no weight)"""
    results = Results([], [], top=5)
    results.add_result({0, 1}, weight=1, trim_to_top=True)
    # When adding to empty results, replaces the initial empty entry
    assert len(results.get_paths) == 1
    assert {0, 1} in [set(p) for p in results.get_paths]


def test_results_only_paths_provided():
    """Test Results raises error when only paths provided without weights"""
    import pytest
    with pytest.raises(ValueError, match="Both paths and weights"):
        Results(paths=[{0}], weights=None, top=1)


def test_results_only_weights_provided():
    """Test Results raises error when only weights provided without paths"""
    import pytest
    with pytest.raises(ValueError, match="Both paths and weights"):
        Results(paths=None, weights=[1.0], top=1)


def test_results_from_dict_basic():
    """Test Results.from_dict with valid dict"""
    data = {0: {'path': [0, 1], 'weight': 2.0},
            1: {'path': [1, 2], 'weight': 3.0}}
    results = Results.from_dict(data, allow_subset=False)
    assert len(results.get_paths) == 2
    assert results.get_weights == [3.0, 2.0]  # Sorted by weight descending


def test_results_from_json_file():
    """Test Results.from_json loading from file"""
    import tempfile
    import os
    paths = [{0, 1}, {1, 2}]
    weights = [2.0, 3.0]
    results = Results(paths, weights, top=2)

    # Save to file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = Path(f.name)
    results.to_json(temp_path)

    try:
        # Load from file
        results2 = Results.from_json(temp_path)
        assert results2.get_paths == results.get_paths
        assert results2.get_weights == results.get_weights
    finally:
        os.unlink(temp_path)


def test_results_weights_without_paths():
    """Test Results with weights but no valid paths"""
    results = Results([set()], [1.0], top=1, allow_subset=True)
    assert len(results.get_paths) == 1


def test_results_remap_path_basic():
    """Test Results.remap_path with basic remapping"""
    paths = [{0, 1}, {1, 2}]
    weights = [2.0, 3.0]
    results = Results(paths, weights, top=2)
    index_map = [1, 0, 2]
    remapped = results.remap_path(index_map)
    # Verify remapping occurred
    assert remapped.get_weights == results.get_weights
    assert len(remapped.get_paths) == 2


def test_results_str_representation():
    """Test Results string representation"""
    paths = [{0, 1}, {1, 2}]
    weights = [2.0, 3.0]
    results = Results(paths, weights, top=2)
    str_repr = str(results)
    assert "Results" in str_repr or str(weights[0]) in str_repr


def test_results_top_property_setter():
    """Test Results.top property setter"""
    paths = [{i} for i in range(10)]
    weights = [float(i) for i in range(10)]
    results = Results(paths, weights, top=5)
    assert len(results.get_paths) == 5
    results.top = 3
    assert len(results.get_paths) == 3
    results.top = 10
    assert len(results.get_paths) == 10


def test_results_add_result_with_bisect_false():
    """Test Results.add_result with bisect=False"""
    paths = [{0}, {1}, {2}]
    weights = [1.0, 2.0, 3.0]
    results = Results(paths, weights, top=5)
    results.add_result({5}, 5.0, trim_to_top=True, bisect=False)
    # Should be added at the beginning (highest weight)
    assert results.best.path == {5}


def test_results_add_results_from_results():
    """Test Results.bulk_add_result with another Results object"""
    paths1 = [{0}, {1}]
    weights1 = [1.0, 2.0]
    results1 = Results(paths1, weights1, top=5)

    paths2 = [{2}, {3}]
    weights2 = [3.0, 4.0]

    results1.bulk_add_result(paths2, weights2)
    assert len(results1.get_paths) == 4


def test_results_best_property():
    """Test Results.best property returns highest weight"""
    paths = [{0}, {1}, {2}]
    weights = [1.0, 3.0, 2.0]
    results = Results(paths, weights, top=5)
    assert results.best.weight == 3.0
    assert results.best.path == {1}


def test_results_get_raw_paths():
    """Test Results.get_raw_paths returns unsorted sets"""
    paths = [{2, 1, 0}, {5, 3}]
    weights = [1.0, 2.0]
    results = Results(paths, weights, top=2)
    raw_paths = results.get_raw_paths
    # Results are sorted by weight (descending), so highest weight comes first
    assert raw_paths[0] == {5, 3}  # weight 2.0
    assert raw_paths[1] == {2, 1, 0}  # weight 1.0


if __name__ == '__main__':
    test_remap_path()
