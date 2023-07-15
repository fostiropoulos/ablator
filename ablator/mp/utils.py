import copy

import numpy as np

from ablator.mp.node_manager import Resource


def _sorted_gpu_util(resources: list[Resource]):
    return np.argsort(
        [
            _resources.gpu_free_mem_arr.max() if len(_resources.gpu_free_mem_arr) else 0
            for _resources in resources
        ]
    )[::-1]


def _sorted_cpu_util(resources: list[Resource]):
    return np.argsort([_resources.cpu_mean_util for _resources in resources])


def _sorted_mem_util(resources: list[Resource]):
    return np.argsort([_resources.mem for _resources in resources])


def _sorted_task_util(resources: list[Resource]):
    return np.argsort([len(_resources.running_tasks) for _resources in resources])


def _sort_node_ips_by_util(resources: dict[str, Resource], eval_gpu: bool):
    node_ips = list(resources.keys())
    resources_list = list(resources.values())
    _node_ips = []

    for _ in range(len(node_ips)):
        usage_lists = []
        if eval_gpu:
            usage_lists.append(_sorted_gpu_util(resources_list))
        usage_lists.append(_sorted_cpu_util(resources_list))
        usage_lists.append(_sorted_mem_util(resources_list))
        usage_lists.append(_sorted_task_util(resources_list))
        np_usage_lists = np.array(usage_lists)
        idxs, values = np.unique(np_usage_lists[:, 0], return_counts=True)
        least_util_idx = idxs[np.argmax(values)]
        _node_ips.append(node_ips[least_util_idx])
        del resources_list[least_util_idx]
        del node_ips[least_util_idx]

    return _node_ips


def _sorted_nodes_by_util(
    resources: dict[str, Resource],
    gpu_util_requirement: int | None = None,
    memory_perc_limit: int = 80,
    cpu_util_perc_limit: int = 80,
) -> list[str]:
    """
    _sorted_nodes_by_util Sorts the nodes based on their available resources from
    the least used to the most used node. If a node does not meet the `gpu_util_requirement` or
    `memory_perc_limit` and `cpu_util_perc_limit` it is excluded from the list.

    Parameters
    ----------
    resources : dict[str, Resource]
        a dictionary of the nodes with their available resources
    gpu_util_requirement : int | None, optional
        the gpu requirement for the task, by default None
    memory_perc_limit : int, optional
        the percentage upper limit to memory utilization, by default 80
    cpu_util_perc_limit : int, optional
        the percentage upper limit to cpu utilization, by default 80

    Returns
    -------
    list[str]
        the sorted list of node ips sorted from the least to most used.
    """

    node_ips = _sort_node_ips_by_util(resources, gpu_util_requirement is not None)

    def _should_sample(node_ip):
        ray_cluster_gpu_limit = gpu_util_requirement is None or any(
            resources[node_ip].gpu_free_mem_arr > gpu_util_requirement
        )
        ray_cluster_cpu_limit = resources[node_ip].cpu_mean_util < cpu_util_perc_limit
        ray_cluster_mem_limit = resources[node_ip].mem < memory_perc_limit
        return ray_cluster_mem_limit and ray_cluster_cpu_limit and ray_cluster_gpu_limit

    _free_nodes = []
    for node_ip in copy.deepcopy(node_ips):
        if _should_sample(node_ip):
            _free_nodes.append(node_ip)
    return _free_nodes
