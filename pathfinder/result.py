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
    def __init__(self, paths:list[set], weights:list[float], top:int=1, ignore_subset:bool=False):
        if len(paths) != len(weights):
            raise ValueError("Unequal length lists provided!")
        self.ignore_subset = ignore_subset
        self._res = self.__set_res(paths, weights)
        self._top = top
        

    #setter
    def __set_res(self, pths:list[set], wghts:list[float], sort:bool=True) -> List[Result]:
        all_res = [Result(path=set(p), weight=w) for p, w in zip(pths, wghts)]
        if self.ignore_subset:
            res = [item for item in all_res if not any(item.path < pth.path for pth in all_res)]
        else:
            res = all_res
        if sort:
            return sorted(res, reverse=True)
        return res

    @property
    def top(self)->None:
        return self._top

    @top.setter
    def top(self, top)->None:
            self._top = top

    @property
    def res(self) -> List[Result]:
        return self._res[:self._top]

    @property
    def get_weights(self) -> List[float]:
        return [item.weight for item in self.res]

    @property
    def get_paths(self) -> List[list]:
        return [sorted(item.path) for item in self.res]

    @property
    def get_raw_paths(self) -> List[list]:
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

    def add_res(self, path:set, weight:float, trim_to_top:bool=True, bisect=True)-> None:
        res_ = Result(path=set(path), weight=weight)
        if self.ignore_subset:
            if any(res_.path < pth for pth in self.get_raw_paths):
                return
        if res_ not in self._res:
            if bisect:
                idx = self.bisect_left(self._res, res_)
                self._res.insert(idx, res_)
            else:
                self._res.append(res_)
        if trim_to_top:
            self._res = self.res

    def bulk_add(self, paths:list[set], weight:list[float])->None:
        if max(weight) > min(self.get_weights):
            if self.ignore_subset:
                for pth, wgt in zip(paths, weight):
                    self.add_res(pth, wgt)
            else:
                self._res += self.__set_res(paths, weight, sort=True)
                self.res_sort(trim=True)

    def remap_path(self, index_map:list)->list[Result]:
        ret = []
        for item in self.res:
            map_path = set([index_map[i] for i in item.path])
            ret.append(Result(path=map_path, weight=item.weight))
        return ret

    def __str__(self):
        return ",\n".join([f"{i+1}: {item}" for i, item in enumerate(self.res)])
