from graph_utils import create_network_graph
from compare import get_run_result


def graphViNE_run(physical_graph, requests, save=True):
    physical_graph_vine = physical_graph.copy()
    request_graphs_graph_vine = [r.copy() for r in requests]
    results = get_run_result(
        physical_graph_vine, request_graphs_graph_vine, method="graphViNE", traffic_load=220, max_time=1500, cost_revenue=True, utils=True)
    

    [gv_probs, gv_blockeds, gv_num, gv_cost, gv_revenue, gv_cpu_utils, gv_link_utils] = results
    return gv_probs, gv_blockeds, gv_num, gv_cost, gv_revenue, gv_cpu_utils, gv_link_utils 



def main():
    physical_graph = create_network_graph(nodes_num=100)
    requests = [create_network_graph(np.random.randint(3, 11), min_feature_val=4, max_feature_val=10,
                                     min_link_val=4, max_link_val=10, connection_prob=0.7, life_time=(100, 900)) for i in range(6000)]

    # physical_grc = physical_graph.copy()
   

if __name__ == "__main__":
    main()
