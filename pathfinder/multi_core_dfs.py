from .dfs import WHDFS
from .result import Results, Result
from .matrix_handler import BinaryAcceptance
from multiprocessing import Process, Manager
import multiprocessing.shared_memory as shm
from functools import partial
import numpy as np
from typing import Dict, Optional


def split_list_into_sublists(lst, n):
    # Create a list of empty sublists
    sublists = [[] for _ in range(n)]

    # Distribute the first n items across the sublists
    for i in range(n):
        if i < len(lst):
            sublists[i].append(lst[i])

    # Distribute the remaining items
    for i in range(n, len(lst)):
        sublists[i % n].append(lst[i])

    return sublists


def shared_memory_value(shared_object_name, num_cor, idx, value):
    existing_shm = shm.SharedMemory(name=shared_object_name)
    shared_array = np.ndarray(num_cor, dtype=np.float64, buffer=existing_shm.buf)
    shared_array[idx] = value
    new_max = max(shared_array)
    existing_shm.close()
    return new_max


def get_best_weight(shared_object_name, num_cor) -> float:
    existing_shm = shm.SharedMemory(name=shared_object_name)
    shared_array = np.ndarray(num_cor, dtype=np.float64, buffer=existing_shm.buf)
    value = max(shared_array)
    existing_shm.close()
    return value


def _whdfs_worker(args: Dict, return_dict: dict[int, Result], shared_object_name: str) -> None:

    result = WHDFS(binary_acceptance_obj=args['bam'], top=args['top'], ignore_subset=args['ignore_subset'])
    result.set_shared_memory_update = partial(shared_memory_value, shared_object_name=shared_object_name,
                                              num_cor=args['num_cor'], idx=args['childId'])
    result.set_top_weight = partial(get_best_weight, shared_object_name=shared_object_name, num_cor=args['num_cor'])
    # result.add_results_from_results(args['result'])
    result.find_paths(runs=None, ignore_child=args['ignore_nodes'], reset_result=False)

    return_dict.update({args['childId']: result})


def run_multicore_hdfs(binary_acceptance_obj: BinaryAcceptance, num_cor: int = 1, top: int = 10,
                       ignore_subset: bool = True, runs: Optional[int] = None):

    args = dict(bam=binary_acceptance_obj, top=top, ignore_subset=ignore_subset, runs=None)
    # result = Results(paths=[{}], weights=[0.0], top=top, ignore_subset=ignore_subset)
    if runs is None or runs > binary_acceptance_obj.dim:
        runs = binary_acceptance_obj.dim
    args['runs'] = 1
    # args['result'] = result
    args['num_cor'] = num_cor
    args['top'] = top
    args['nbytes'] = np.zeros(num_cor, dtype=np.float64).nbytes
    manager = Manager()
    outputdict = manager.dict()
    jobs = []
    shm_object = shm.SharedMemory(name='top_results', create=True, size=args['nbytes'])
    chunked = {i: [] for i in range(num_cor)}
    for source in range(0, runs):
        binary_acceptance_obj.reset_source(source=source)
        available = binary_acceptance_obj.get_source_row_index
        for i, run_chunk in enumerate(split_list_into_sublists(available, num_cor)):
            chunked[i].append([j for j in available if j not in run_chunk])
    for child, index_list in chunked.items():
        args['ignore_nodes'] = index_list
        args['childId'] = child
        p = Process(target=_whdfs_worker, args=(args, outputdict, shm_object.name))
        jobs.append(p)
        p.start()
    for p in jobs:
        p.join()
    result = Results(paths=[{}], weights=[0.0], top=top, ignore_subset=ignore_subset)
    for _, res in dict(outputdict).items():
        result.add_results_from_results(res)
    shm_object.close()
    shm_object.unlink()

    return result
