from graph_utils import create_network_graph
from compare import get_run_result
import numpy as np
from result_utils import save_data


def graphViNE_run(physical_graph, requests, save=True):
    physical_graph_vine = physical_graph.copy()
    request_graphs_graph_vine = [r.copy() for r in requests]
    results = get_run_result(
        physical_graph_vine, request_graphs_graph_vine, method="graphViNE", traffic_load=2000, max_time=500, cost_revenue=True, utils=True)
    
    if save:
        names = ['gv_probs', 'gv_blockeds', 'gv_num', 'gv_cost', 'gv_revenue', 'gv_cpu_utils', 'gv_link_utils']
        for i in range(len(results)):
            save_data(results[i], names[i])
    [gv_probs, gv_blockeds, gv_num, gv_cost, gv_revenue, gv_cpu_utils, gv_link_utils] = results
    return gv_probs, gv_blockeds, gv_num, gv_cost, gv_revenue, gv_cpu_utils, gv_link_utils 


def neuroViNE_run(physical_graph, requests, save=True):
    physical_neuro = physical_graph.copy()
    request_graphs_neuro = [r.copy() for r in requests]
    results = get_run_result(
        physical_neuro, request_graphs_neuro, method="neuroViNE", traffic_load=2000, max_time=500, cost_revenue=True, utils=True)
    
    if save:
        names = ['nv_probs', 'nv_blockeds', 'nv_num', 'nv_cost', 'nv_revenue', 'nv_cpu_utils', 'nv_link_utils']
        for i in range(len(results)):
            save_data(results[i], names[i])
    [nv_probs, nv_blockeds, nv_num, nv_cost, nv_revenue, nv_cpu_utils, nv_link_utils] = results
    return nv_probs, nv_blockeds, nv_num, nv_cost, nv_revenue, nv_cpu_utils, nv_link_utils 

def main():
    physical_graph = create_network_graph(nodes_num=100)
    requests = [create_network_graph(np.random.randint(3, 11), min_feature_val=4, max_feature_val=10,
                                     min_link_val=4, max_link_val=10, connection_prob=0.7, life_time=(100, 900)) for i in range(6000)]

    # graphViNE_run(physical_graph, requests)
    neuroViNE_run(physical_graph, requests)

if __name__ == "__main__":
    main()
