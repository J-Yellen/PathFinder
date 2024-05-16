from .dfs import WHDFS, HDFS
from .result import Results, Result
from .matrix_handler import BinaryAcceptance
from multiprocessing import Process, Manager
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


def _hdfs_worker(args: Dict, return_dict: dict[int, Result]) -> None:
    """
    Multi-processing worker for find_best_sets

    Args:
        pseudo_gen_dicts (List[Dict]): List of dictionaries containing a binary acceptance matrix
                                       and set of corresponding weights.
        run_num (int): Unique integer identifier for labeling return dictionary
        return_dict (Dict): DictProxy for Manager
    """
    hdfs = WHDFS if args['weighted'] else HDFS
    result = hdfs(binary_acceptance_obj=args['bam'], top=args['top'], ignore_subset=args['ignore_subset'])
    current = args.get('result', False)
    if current:
        result.add_results_from_results(current)
    result.find_paths(runs=args['runs'], source_node=args['source'], ignore_child=args['ignore_nodes'])
    return_dict.update({args['childId']: result})


def run_multicore_hdfs(binary_acceptance_obj: BinaryAcceptance, num_cor: int = 1, top: int = 10,
                       weighted: bool = True, ignore_subset: bool = True, runs: Optional[int] = None):

    args = dict(bam=binary_acceptance_obj, weighted=weighted, top=top, ignore_subset=ignore_subset, runs=None)
    result = Results(paths=[{}], weights=[0.0], top=top, ignore_subset=ignore_subset)
    if runs is None or runs > binary_acceptance_obj.dim:
        runs = binary_acceptance_obj.dim
    args['runs'] = 1
    args['result'] = result
    manager = Manager()
    outputdict = manager.dict()
    for source in range(0, runs):
        binary_acceptance_obj.reset_source(source=source)
        available = binary_acceptance_obj.get_source_row_index
        if binary_acceptance_obj.get_weight(list(available)) < result.best.weight:
            continue
        chunked = split_list_into_sublists(available, num_cor)
        args['source'] = source
        for child, index_list in enumerate(chunked):
            args['ignore_nodes'] = [j for j in available if j not in index_list]
            jobs = []
            for child in range(num_cor):
                args['childId'] = child
                p = Process(target=_hdfs_worker, args=(args, outputdict))
                jobs.append(p)
                p.start()
            for p in jobs:
                p.join()
            for _, res in outputdict.items():
                result.add_results_from_results(res)
            args['result'] = result

    return result
