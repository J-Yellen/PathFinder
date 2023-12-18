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
    res2 = Result(path=path, weight=weight-1)
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


def test_results_add_res():
    paths = [sim_path(max_value=20) for _ in range(100)]
    weights = [sum(p) for p in paths]
    results = Results(paths, weights, top=10)
    results.add_res({1, 2, 3, 4}, 1000)
    assert results.best.weight == 1000


if __name__ == '__main__':
    test_results_class()
    test_results_add_res()
