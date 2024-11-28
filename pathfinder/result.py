"""
#####################################
# Part of the PATH FINDER  Module   #
# Author J.Yellen                   #
#####################################
"""
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple
from os import PathLike
import json


@dataclass(order=True)
class Result:
    """
    Data Class for individual path and associated weight
    sort index = weight for comparison
    """
    sort_index: float = field(init=False, repr=False)
    path: set
    weight: float

    def __post_init__(self) -> None:
        self.sort_index = self.weight

    def __repr__(self) -> str:
        return f"Path = {sorted(self.path)},  Weight = {self.weight}"


class Results():
    """
    Results class to handle lists of Result (path, weight) objects
    """
    def __init__(self, paths: Optional[list[set]] = None, weights: Optional[list[float]] = None, top: int = 1,
                 allow_subset: bool = False):
        self.allow_subset = allow_subset
        self._top = top
        if paths is None:
            self._res = []
        else:
            if len(paths) != len(weights):
                raise ValueError("Unequal length lists provided!")
            self._res = self._set_res(paths, weights)

    @classmethod
    def from_dict(self, result_dict: dict[int, dict], allow_subset: bool = False) -> 'Results':
        top = len(result_dict)
        paths = [set(item.get('path')) for _, item in result_dict.items()]
        weights = [item.get('weight') for _, item in result_dict.items()]
        return Results(paths=paths, weights=weights, top=top, allow_subset=allow_subset)

    @classmethod
    def from_json(self, json_file_name: PathLike, allow_subset: bool = False) -> 'Results':
        with open(json_file_name, "r") as json_file:
            result_dict = json.load(json_file)
        return self.from_dict(result_dict, allow_subset=allow_subset)

    def to_dict(self) -> dict:
        return {f"{i}": {'path': list(item.path), 'weight': item.weight} for i, item in enumerate(self.res)}

    def to_json(self, file_name: PathLike) -> None:
        with open(file_name, "w") as outfile:
            json.dump(self.to_dict(), outfile, indent=4)

    def _set_res(self, paths: list[set], weights: list[float], sort: bool = True) -> Tuple[list[Result], bool]:
        """
        Takes lists of paths and weights and combines to create a list of Result type objects.

        Args:
            paths (list[set]): List of unique sets containing index reference of directed acyclic graph
            weights (list[float]): Weights corresponding to the path input
            sort (bool, optional): Sort by path weight. Defaults to True.

        Returns:
            list[Result]: List of result objects containing result for each path input
        """
        all_res = [Result(path=set(p), weight=w) for p, w in zip(paths, weights)]
        if not self.allow_subset:
            res = [item for item in all_res if not any(item.path < pth.path for pth in all_res)]
        else:
            res = all_res
        if sort:
            res = sorted(res, reverse=True)
        return res

    @property
    def top(self) -> None:
        return self._top

    @top.setter
    def top(self, top) -> None:
        self._top = top

    @property
    def res(self) -> list[Result]:
        if self._res:
            return self._res[:self._top]
        else:
            return [Result({None}, 0)]

    @property
    def get_weights(self) -> list[float]:
        return [item.weight for item in self.res]

    @property
    def get_paths(self) -> list[set]:
        return [sorted(item.path) for item in self.res]

    @property
    def get_raw_paths(self) -> list[set]:
        return [item.path for item in self.res]

    @property
    def best(self) -> Result:
        return max(self.res)

    @staticmethod
    def _bisect_left(to_bisect: list[Result], new_item: Result,
                     lo_: int = 0, hi_: Optional[int] = None) -> int:
        """
        Calculates the index position of new item that is to be inserted in a descending
        list i.e. inserted to the left.

        Args:
            to_bisect (list[Result]): Original list of sortable objects
            new_item (Result): object to be placed in original list
            lo_ (int, optional): Lower bound from which to sort new item. Defaults to 0.
            hi_ (int, optional): Upper bound to which to sort new item. Defaults as None == length list.

        Returns:
            int: index for insertion to left of descending values
        """
        if hi_ is None:
            hi_ = len(to_bisect)
        while lo_ < hi_:
            mid = (lo_ + hi_) // 2
            if to_bisect[mid].weight >= new_item.weight:
                lo_ = mid + 1
            else:
                hi_ = mid
        return lo_

    def res_sort(self, trim: bool = True) -> list[Result]:
        if trim:
            self._res = sorted(self._res, reverse=True)[:self._top]
        self._res = sorted(self._res, reverse=True)

    def add_result(self, path: set, weight: float, trim_to_top: bool = True, bisect=True) -> None:
        """
        Add new result to list of results in Path, weight format

        Args:
            path (set): Sets containing index reference of directed acyclic graph
            weight (float): Weights corresponding to the path input
            trim_to_top (bool, optional): Restrict list of results to default length. Defaults to True.
            bisect (bool, optional): Insert new result into sorted Results position. Defaults to True.
        """
        res_ = Result(path=set(path), weight=weight)
        add_result = True
        if not self.allow_subset:
            add_result = not any(res_.path <= pth for pth in self.get_raw_paths)
        if add_result:
            if res_ not in self._res:
                if bisect:
                    idx = self._bisect_left(self._res, res_)
                    self._res.insert(idx, res_)
                else:
                    self._res.append(res_)
            if trim_to_top:
                self._res = self.res

    def add_results_from_results(self, result: 'Results') -> None:
        self.bulk_add_result(result.get_paths, result.get_weights)

    def bulk_add_result(self, paths: list[set], weight: list[float]) -> None:
        """
        Add new multiple new results to list of results as lists of paths and weights.
        Args:
            paths (list[set]): Lists of Sets containing index reference of directed acyclic graph
            weight (list[float]): List of Weights corresponding to the path input
        """
        if max(weight) > min(self.get_weights):
            if not self.allow_subset:
                for pth, wgt in zip(paths, weight):
                    self.add_result(pth, wgt)
            else:
                new = self._set_res(paths, weight, sort=True)
                self._res.extend(new)
                self.res_sort(trim=True)

    def remap_path(self, index_map: Optional[Union[dict, list]] = None, weight_offset: float = 0.0) -> 'Results':
        """
        Convert result path index using ether a dictionary or list of mapped indices.

        Args:
            index_map (Union[dict, list]): Iterable Dictionary or list containing new index
            values as ether:
            - List[ Current_Index ] = New_Index

            or
            - Dict[ Current_Index ] = New_Index

        Returns:
            list[dict[list, float]]:  List of results in dictionary format
            that preserved the new path map order
        """

        dict_of_results = {}
        if index_map is None:
            for i, item in enumerate(self.res):
                dict_of_results[i] = {'path': {p for p in sorted(item.path)},
                                      'weight': item.weight - (len(item.path) * weight_offset)}

        else:
            for i, item in enumerate(self.res):
                dict_of_results[i] = {'path': {int(index_map[p]) for p in sorted(item.path)},
                                      'weight': item.weight - (len(item.path) * weight_offset)}

        return self.from_dict(dict_of_results, allow_subset=self.allow_subset)

    def __eq__(self, other: 'Results') -> bool:
        equal = False
        if self.top == other.top:
            other_paths_in_self = all([o.path == s.path for o, s in zip(other.res, self.res)])
            weights_match = self.get_weights == other.get_weights
            equal = other_paths_in_self & weights_match
        return equal

    def __str__(self):
        return ",\n".join([f"{i+1}: {item}" for i, item in enumerate(self.res)])
