"""
#####################################
# Part of the PATH FINDER  Module   #
# Author J.Yellen                   #
#####################################
"""
from dataclasses import dataclass, field
from typing import List

@dataclass(order=True)
class Result:
    """
    Data Class for individual path and associated weight
    sort index = weight for comparison
    """
    sort_index: float = field(init=False, repr=False)
    path: list
    weight: float

    def __post_init__(self) -> None:
        self.sort_index = self.weight

    def __repr__(self) -> str:
        return f"Path = {self.path},  Weight = {self.weight}"

class Results():
    """
    Results class to handle lists of Result (path, weight) objects
    """
    def __init__(self, paths:list[list], weights:list[float], top:int|None=None):
        if len(paths) != len(weights):
            raise ValueError("Unequal length lists provided!")
        self._res = self.__set_res(paths, weights)
        self._top = self.__set_top(top)

    #setter
    @staticmethod
    def __set_res(pths:list[list], wghts:list[float], sort:bool=True) -> List[Result]:
        res = [Result(path=p, weight=w) for p, w in zip(pths, wghts)]
        if sort:
            return sorted(res, reverse=True)
        return res

    #setter
    def __set_top(self, top:int)->int:
        if top is None:
            return len(self._res)
        return int(top)

    @property
    def res(self) -> List[Result]:
        return self._res[:self._top]

    @property
    def get_weights(self) -> List[float]:
        return [item.weight for item in self.res]

    @property
    def get_paths(self) -> List[list]:
        return [item.path for item in self.res]

    @property
    def best(self) -> Result:
        return max(self._res)

    @staticmethod
    def bisect_left(to_bisect:list, num:object, lo_:int=0, hi_:int=None)->int:
        if hi_ is None:
            hi_ = len(to_bisect)
        while lo_ < hi_:
            mid = (lo_ + hi_)//2
            if to_bisect[mid] > num:
                lo_ = mid + 1
            else:
                hi_ = mid
        return lo_

    def res_sort(self, trim:bool=True)->List[Result]:
        if trim:
            self._res = sorted(self._res, reverse=True)[:self._top]
        self._res = sorted(self._res, reverse=True)

    def add_res(self, path:list, weight:float)-> None:
        res_ = Result(path=path, weight=weight)
        idx = self.bisect_left(self._res, res_)
        # if idx == 0 and weight == self.best.weight:
        #     idx += 1
        self._res.insert(idx, res_)
        self._res = self.res

    def bulk_add(self, paths:list[list], weight:list[float])->None:
        if max(weight) > min(self.get_weights):
            #self._res = self.__set_res(paths + self.get_paths, weight + self.get_weights)
            self._res += self.__set_res(paths, weight, sort=False)
            self.res_sort(trim=True)

    def reset_top(self, new_top:int)-> None:
        self.__set_top(new_top)
        self.res_sort()

    def __str__(self):
        return ",\n".join([f"{i+1}: {item}" for i, item in enumerate(self.res)])
