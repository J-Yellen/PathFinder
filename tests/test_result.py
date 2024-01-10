import numpy as np
from pathfinder.result import Result, Results

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
    weights = [sum(p) for p in paths]
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
    new_item1 = Result({5.5}, 5.5)
    new_item2 = Result({5.1}, 5)
    new_item3 = Result({10}, 10.0)
    idx1 = results._bisect_left(results._res, new_item1)
    idx2 = results._bisect_left(results._res, new_item2)
    idx3 = results._bisect_left(results._res, new_item3)
    assert idx1 == 4
    assert idx2 == 5
    assert idx3 == 0


def test_add_result():
    paths = [{i} for i in range(10)]
    weights = [float(i) for i in range(10)]
    results = Results(paths, weights, top=10)
    results.add_result({5.1}, 5.0, trim_to_top=True, bisect=True)
    results.add_result({10}, 10, trim_to_top=True, bisect=True)
    assert results.res[6] == Result({5.1}, 5.0)
    assert results.res[0] == Result({10}, 10)


def test_bulk_add_res():
    paths = [sim_path(max_value=20) for _ in range(100)]
    weights = [sum(p) for p in paths]
    paths2 = [sim_path(max_value=20) for _ in range(100)] + [{0, 1000}]
    weights2 = [sum(p) for p in paths] + [1000]
    results = Results(paths, weights, top=10)
    results.bulk_add_result(paths2, weights2)
    assert len(results.res) == 10
    assert results.best == Result({0, 1000}, 1000)


def test_remap_path():

    paths = [{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 5, 9}]
    results = Results(paths, [1, 1], top=2)
    map1 = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    map2 = dict(enumerate(map1))
    map_result1 = results.remap_path(map1)
    map_result2 = results.remap_path(map2)
    assert map_result1 == map_result2
    assert map_result1[1]['Path'] == {map1[i] for i in paths[1]}


if __name__ == '__main__':
    test_remap_path()
