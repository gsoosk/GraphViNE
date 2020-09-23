import sys
from graph_utils import create_network_graph
from compare import get_run_result
import numpy as np
from result_utils import save_data
import _thread as thread
import time


def save_results(model_name, results):
    if len(results) == 7:
        names = [f'{model_name}_probs', f'{model_name}_blockeds', f'{model_name}_num', f'{model_name}_cost',
                 f'{model_name}_revenue', f'{model_name}_cpu_utils', f'{model_name}_link_utils']
    else:
        names = [f'{model_name}_probs', f'{model_name}_blockeds', f'{model_name}_num', f'{model_name}_cost',
                 f'{model_name}_revenue', f'{model_name}_cpu_utils', f'{model_name}_link_utils',
                 f'{model_name}_gpu_utils', f'{model_name}_memory_utils']
    for i in range(len(results)):
        save_data(results[i], names[i])


def graphViNE_run(physical_graph, requests, save=True, load=2000, max_time=500, verbose=True, n_clusters = 4):
    physical_graph_vine = physical_graph.copy()
    request_graphs_graph_vine = [r.copy() for r in requests]
    results = get_run_result(
        physical_graph_vine, request_graphs_graph_vine, method="graphViNE", traffic_load=load, max_time=max_time, cost_revenue=True, utils=True, verbose=verbose, 
        n_clusters=n_clusters)

    if save:
        save_results(f'gv{n_clusters}', results)

    return results


def neuroViNE_run(physical_graph, requests, save=True, load=2000, max_time=500, verbose=True):
    physical_neuro = physical_graph.copy()
    request_graphs_neuro = [r.copy() for r in requests]
    results = get_run_result(
        physical_neuro, request_graphs_neuro, method="neuroViNE", traffic_load=load, max_time=max_time, cost_revenue=True, utils=True, verbose=verbose)

    if save:
        save_results('nv', results)

    return results


def grc_run(physical_graph, requests, save=True, load=2000, max_time=500, verbose=True):
    physical_grc = physical_graph.copy()
    request_graphs_grc = [r.copy() for r in requests]
    results = get_run_result(
        physical_grc, request_graphs_grc, method="grc", traffic_load=load, max_time=max_time, cost_revenue=True, utils=True,  verbose=verbose)

    if save:
        save_results('grc', results)

    return results


def best_fit_run(physical_graph, requests, save=True, load=2000, max_time=500, verbose=True):
    physical_best_fit = physical_graph.copy()
    request_graphs_best_fit = [r.copy() for r in requests]
    results = get_run_result(
        physical_best_fit, request_graphs_best_fit, method="bestFit", traffic_load=load, max_time=max_time, cost_revenue=True, utils=True, verbose=verbose)

    if save:
        save_results('bf', results)

    return results


def first_fit_run(physical_graph, requests, save=True, load=2000, max_time=500, verbose=True):
    physical_first_fit = physical_graph.copy()
    request_graphs_first_fit = [r.copy() for r in requests]
    results = get_run_result(
        physical_first_fit, request_graphs_first_fit, method="firstFit", traffic_load=load, max_time=max_time, cost_revenue=True, utils=True, verbose=verbose)

    if save:
        save_results('ff', results)

    return results


def compute():
    np.random.seed(64)  # to get a unique result every time
    physical_graph = create_network_graph(nodes_num=100)
    requests = [create_network_graph(np.random.randint(3, 11), min_feature_val=4, max_feature_val=10,
                                     min_link_val=4, max_link_val=10, connection_prob=0.7, life_time=(100, 900)) for i in range(12500)]

    load = 1000
    max_time = 2000
    
    graphViNE_run(physical_graph, requests, load=load,
                  max_time=max_time, verbose=False, n_clusters=5)
    neuroViNE_run(physical_graph, requests, load=load,
                  max_time=max_time, verbose=True)
    grc_run(physical_graph, requests, load=load,
            max_time=max_time, verbose=True)
    best_fit_run(physical_graph, requests, load=load,
                 max_time=max_time, verbose=True)
    first_fit_run(physical_graph, requests, load=load,
                  max_time=max_time, verbose=True)


def compare():
    folder = './results/test'
    grc_probs = np.fromfile(f'{folder}/grc_probs.dat')
    nv_probs = np.fromfile(f'{folder}/nv_probs.dat')
    gv_probs = np.fromfile(f'{folder}/gv_probs.dat')
    bf_probs = np.fromfile(f'{folder}/bf_probs.dat')
    ff_probs = np.fromfile(f'{folder}/ff_probs.dat')

    from result_utils import draw_blocking_prob
    draw_blocking_prob([grc_probs, gv_probs, bf_probs, ff_probs, nv_probs],
                       ['GRC', 'GraphViNE', 'Best Fit', 'First Fit', 'NeuroViNE'],
                       ['r', 'b', 'purple', 'green', 'y'],
                       'Time Units', 'Blocking prob', 'Models Block Probability with 2 Request Per Time Unit - (VNRs Link max = 50 instead of 10)',
                       name='bp', scale=4)

    draw_blocking_prob([1 - grc_probs, 1 - gv_probs, 1 - bf_probs, 1 - ff_probs, 1 - nv_probs],
                       ['GRC', 'GraphViNE', 'Best Fit', 'First Fit', 'NeuroViNE'],
                       ['r', 'b', 'purple', 'green', 'y'],
                       'Time Units', 'Acceptance Ratio', 'Models Acceptance Ratio with 2 Request Per Time Unit - (VNRs Link max = 50 instead of 10)',
                       name='ar', scale=4)

    grc_revenue = sum(np.fromfile(f'{folder}/grc_revenue.dat', dtype=int))
    nv_revenue = sum(np.fromfile(f'{folder}/nv_revenue.dat', dtype=int))
    gv_revenue = sum(np.fromfile(f'{folder}/gv_revenue.dat', dtype=int))
    bf_revenue = sum(np.fromfile(f'{folder}/bf_revenue.dat', dtype=int))
    ff_revenue = sum(np.fromfile(f'{folder}/ff_revenue.dat', dtype=int))

    from result_utils import draw_bars
    draw_bars(['GRC', 'GraphViNE', 'Best Fit', 'First Fit', 'NeuroViNE'],
              [grc_revenue, gv_revenue, bf_revenue, ff_revenue, nv_revenue],
              '2000 Time Units - 2 req per tt - VNRs max link 40',
              name='revenue')

    grc_cost = sum(np.fromfile(f'{folder}/grc_cost.dat', dtype=int))
    nv_cost = sum(np.fromfile(f'{folder}/nv_cost.dat', dtype=int))
    gv_cost = sum(np.fromfile(f'{folder}/gv_cost.dat', dtype=int))
    bf_cost = sum(np.fromfile(f'{folder}/bf_cost.dat', dtype=int))
    ff_cost = sum(np.fromfile(f'{folder}/ff_cost.dat', dtype=int))

    from result_utils import draw_bars
    draw_bars(['GRC', 'GraphViNE', 'Best Fit', 'First Fit', 'NeuroViNE'],
              [grc_cost, gv_cost, bf_cost, ff_cost, nv_cost],
              '2000 Time Units - 2 req per tt - VNRs max link 40',
              name='cost')

    grc_cpu_util = -sum(np.fromfile(f'{folder}/grc_cpu_utils.dat')) / ((np.fromfile(
        f'{folder}/grc_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/grc_num.dat', dtype=int)[0])
    nv_cpu_util = -sum(np.fromfile(f'{folder}/nv_cpu_utils.dat')) / ((np.fromfile(
        f'{folder}/nv_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/nv_num.dat', dtype=int)[0])
    gv_cpu_util = -sum(np.fromfile(f'{folder}/gv_cpu_utils.dat')) / ((np.fromfile(
        f'{folder}/gv_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/gv_num.dat', dtype=int)[0])
    bf_cpu_util = -sum(np.fromfile(f'{folder}/bf_cpu_utils.dat')) / ((np.fromfile(
        f'{folder}/bf_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/bf_num.dat', dtype=int)[0])
    ff_cpu_util = -sum(np.fromfile(f'{folder}/ff_cpu_utils.dat')) / ((np.fromfile(
        f'{folder}/ff_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/ff_num.dat', dtype=int)[0])

    from result_utils import draw_bars
    draw_bars(['GRC', 'GraphViNE', 'Best Fit', 'First Fit', 'NeuroViNE'],
              [grc_cpu_util, gv_cpu_util, bf_cpu_util, ff_cpu_util, nv_cpu_util],
              '2000 Time Units - 2 req per tt - VNRs max link 40',
              name='cpu_util')

    grc_link_util = -sum(np.fromfile(f'{folder}/grc_link_utils.dat')) / ((np.fromfile(
        f'{folder}/grc_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/grc_num.dat', dtype=int)[0])
    nv_link_util = -sum(np.fromfile(f'{folder}/nv_link_utils.dat')) / ((np.fromfile(
        f'{folder}/nv_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/nv_num.dat', dtype=int)[0])
    gv_link_util = -sum(np.fromfile(f'{folder}/gv_link_utils.dat')) / ((np.fromfile(
        f'{folder}/gv_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/gv_num.dat', dtype=int)[0])
    bf_link_util = -sum(np.fromfile(f'{folder}/bf_link_utils.dat')) / ((np.fromfile(
        f'{folder}/bf_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/bf_num.dat', dtype=int)[0])
    ff_link_util = -sum(np.fromfile(f'{folder}/ff_link_utils.dat')) / ((np.fromfile(
        f'{folder}/ff_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/ff_num.dat', dtype=int)[0])

    from result_utils import draw_bars
    draw_bars(['GRC', 'GraphViNE', 'Best Fit', 'First Fit', 'NeuroViNE'],
              [grc_link_util, gv_link_util, bf_link_util,
                  ff_link_util, nv_link_util],
              '2000 Time Units - 2 req per tt - VNRs max link 40',
              name='link_util')

def compare_extra():
    folder = './results/test'
    gv_probs = np.fromfile(f'{folder}/gv_probs.dat')
    bf_probs = np.fromfile(f'{folder}/bf_probs.dat')
    ff_probs = np.fromfile(f'{folder}/ff_probs.dat')

    from result_utils import draw_blocking_prob
    draw_blocking_prob([gv_probs, bf_probs, ff_probs],
                       ['GraphViNE', 'Best Fit', 'First Fit'],
                       ['b', 'purple', 'green'],
                       'Time Units', 'Blocking prob', 'Models Block Probability with 2 Request Per Time Unit - (VNRs Link max = 50 instead of 10)',
                       name='bp', scale=4)

    draw_blocking_prob([1 - gv_probs, 1 - bf_probs, 1 - ff_probs],
                       ['GraphViNE', 'Best Fit', 'First Fit'],
                       ['b', 'purple', 'green'],
                       'Time Units', 'Acceptance Ratio', 'Models Acceptance Ratio with 2 Request Per Time Unit - (VNRs Link max = 50 instead of 10)',
                       name='ar', scale=4)

    gv_revenue = sum(np.fromfile(f'{folder}/gv_revenue.dat', dtype=int))
    bf_revenue = sum(np.fromfile(f'{folder}/bf_revenue.dat', dtype=int))
    ff_revenue = sum(np.fromfile(f'{folder}/ff_revenue.dat', dtype=int))

    from result_utils import draw_bars
    draw_bars(['GraphViNE', 'Best Fit', 'First Fit'],
              [gv_revenue, bf_revenue, ff_revenue],
              '2000 Time Units - 2 req per tt - VNRs max link 40',
              name='revenue')

    gv_cost = sum(np.fromfile(f'{folder}/gv_cost.dat', dtype=int))
    bf_cost = sum(np.fromfile(f'{folder}/bf_cost.dat', dtype=int))
    ff_cost = sum(np.fromfile(f'{folder}/ff_cost.dat', dtype=int))

    from result_utils import draw_bars
    draw_bars(['GraphViNE', 'Best Fit', 'First Fit'],
              [gv_cost, bf_cost, ff_cost],
              '2000 Time Units - 2 req per tt - VNRs max link 40',
              name='cost')



    gv_cpu_util = -sum(np.fromfile(f'{folder}/gv_cpu_utils.dat')) / ((np.fromfile(
        f'{folder}/gv_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/gv_num.dat', dtype=int)[0])
    bf_cpu_util = -sum(np.fromfile(f'{folder}/bf_cpu_utils.dat')) / ((np.fromfile(
        f'{folder}/bf_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/bf_num.dat', dtype=int)[0])
    ff_cpu_util = -sum(np.fromfile(f'{folder}/ff_cpu_utils.dat')) / ((np.fromfile(
        f'{folder}/ff_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/ff_num.dat', dtype=int)[0])

    from result_utils import draw_bars
    draw_bars(['GraphViNE', 'Best Fit', 'First Fit'],
              [ gv_cpu_util, bf_cpu_util, ff_cpu_util],
              '2000 Time Units - 2 req per tt - VNRs max link 40',
              name='cpu_util')


    gv_link_util = -sum(np.fromfile(f'{folder}/gv_link_utils.dat')) / ((np.fromfile(
        f'{folder}/gv_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/gv_num.dat', dtype=int)[0])
    bf_link_util = -sum(np.fromfile(f'{folder}/bf_link_utils.dat')) / ((np.fromfile(
        f'{folder}/bf_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/bf_num.dat', dtype=int)[0])
    ff_link_util = -sum(np.fromfile(f'{folder}/ff_link_utils.dat')) / ((np.fromfile(
        f'{folder}/ff_blockeds.dat', dtype=int)[0]) - np.fromfile(f'{folder}/ff_num.dat', dtype=int)[0])

    from result_utils import draw_bars
    draw_bars(['GraphViNE', 'Best Fit', 'First Fit'],
              [gv_link_util, bf_link_util,
                  ff_link_util],
              '2000 Time Units - 2 req per tt - VNRs max link 40',
              name='link_util')


def compute_extra():
    np.random.seed(64)  # to get a unique result every time
    physical_graph = create_network_graph(nodes_num=100, extra_features=True)
    requests = [create_network_graph(np.random.randint(3, 11), min_feature_val=4, max_feature_val=10,
                                     min_link_val=4, max_link_val=10, min_GPU=9, max_GPU=30,
                                     min_mem=4, max_mem=10, extra_features=True, connection_prob=0.7, life_time=(100, 900)) for i in range(12500)]

    load = 1000
    max_time = 2000
    graphViNE_run(physical_graph, requests, load=load,
                  max_time=max_time, verbose=True)
    best_fit_run(physical_graph, requests, load=load,
                 max_time=max_time, verbose=True)
    first_fit_run(physical_graph, requests, load=load,
                  max_time=max_time, verbose=True)

def get_diff_req():
    np.random.seed(64)  # to get a unique result every time
    mean_req_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    times = []
    for i, size in enumerate(mean_req_sizes):
        print(f'Start for size of {size} ({i}/{len(mean_req_sizes)})')

        physical_graph = create_network_graph(nodes_num=100)
        requests = [create_network_graph(np.random.randint(3, size), min_feature_val=4, max_feature_val=10,
                                     min_link_val=4, max_link_val=10, connection_prob=0.7, life_time=(100, 900)) for i in range(2500)]

        load = 1000
        max_time = 51
        ts = time.time()
        graphViNE_run(physical_graph, requests, load=load,
                        max_time=max_time, verbose=False, save=False)
        te = time.time()
        times.append(te - ts)
        print(f'Size of {size} finished in {(te - ts)/102}')
    print(times)
    save_data(np.array(times), 'times')

commands = ['--help', '-h', '--get_results', '--compare_results',
            '--get_results_extra', '--compare_results_extra', '--get_diff_requests_result']
help = '''
    Oprtions:
        --help / -h : shows this help!
        --get_resutls : computes the results of models and store in the ./results repo.
        --compare_results : compares the results using matplotlib
        --get_resutls_extra : compute the results of models with gpu and memory as extra 
                        feature an store in the ./results repo.
        --compare_results_extra : compares the results using matplotlib
        --get_diff_requests_result: get the computation time of graphViNE model with different
                        maximum size of request graphs and store them in the ./results repo.
        '''
if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        print('Undefiend Options')
        print(help)
    elif sys.argv[1] == commands[0] or sys.argv[1] == commands[1]:
        print(help)
    elif sys.argv[1] == commands[2]:
        compute()
    elif sys.argv[1] == commands[3]:
        compare()
    elif sys.argv[1] == commands[4]:
        compute_extra()
    elif sys.argv[1] == commands[5]:
        compare_extra()
    elif sys.argv[1] == commands[6]:
        get_diff_req()
