from model import cluster_using_argva
from grc import grc_embed
from graph_utils import from_networkx, create_network_graph
import torch
from sklearn.cluster import KMeans
from graphViNE_best_fit import graphViNE_embed, free_embedded_request
from compare_utils import compute_cost, compute_utils, compute_revenue
import numpy as np
from neuro_vine import neuro_vine_embed
from best_fit import best_fit_embed
from first_fit import first_fit_embed


def get_run_result(physical, request_graphs, method="graphViNE", max_time=3000,
                   traffic_load=150, avg_life_time=500, verbose=True, cost_revenue=False, utils=False):
    r"""
    traffic load is in erlang
    """
    blockeds = 0
    num = 1
    probs = []
    embeddeds = []
    load_acc = 0
    revenues = []
    costs = []
    request_index = 0
    link_utils = []
    cpu_utils = []
    gpu_utils = []
    memory_utils = []
    pred = []
    model = None
    for t in range(1, max_time):  # Loop over all times
        # compute number of request recieves on this time slot
        load_acc += (traffic_load / avg_life_time)
        req_n_in_this_time = int(load_acc)
        load_acc -= req_n_in_this_time

        for i in range(req_n_in_this_time):
            ##################################### GraphVine Method ################################################
            if method == 'graphViNE':
                if request_index % 50 == 0:  # retrain or not
                    # Change data to torch_geometric data
                    data = from_networkx(physical, normalize=True)
                    data.edge_attr = data.edge_attr / data.edge_attr.sum()
                    device = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')
                    data = data.to(device)
                    # train model
                    model = cluster_using_argva(data, verbose=False, max_epoch=100) if model is None else cluster_using_argva(
                        data, verbose=False, max_epoch=5, pre_trained_model=model)
                    with torch.no_grad():
                        z = model.encode(
                            data.x, data.edge_index, data.edge_attr)
                    pred = KMeans(n_clusters=4).fit_predict(z.cpu().data)

                request_embedded, physical, request_graphs[request_index] = graphViNE_embed(
                    physical, pred, request_graphs[request_index], verbose=False)
            ################################### GRC Method ##########################################################
            elif method == 'grc':
                request_embedded, physical, request_graphs[request_index] = grc_embed(
                    physical, request_graphs[request_index], verbose=False)
            ################################### NeuroViNE Method ##########################################################
            elif method == 'neuroViNE':
                request_embedded, physical, request_graphs[request_index] = neuro_vine_embed(
                    physical, request_graphs[request_index], verbose=False)
            ################################### Best Fit Method ##########################################################
            elif method == 'bestFit':
                request_embedded, physical, request_graphs[request_index] = best_fit_embed(
                    physical, request_graphs[request_index], verbose=False)
            ################################### First Fit Method ##########################################################
            elif method == 'firstFit':
                request_embedded, physical, request_graphs[request_index] = first_fit_embed(
                    physical, request_graphs[request_index], verbose=False)
            num += 1

            if not request_embedded:
                blockeds += 1
            if request_embedded:
                embeddeds.append(
                    [t, request_graphs[request_index], request_index])
                if cost_revenue:
                    revenues.append(compute_revenue(
                        request_graphs[request_index]))
                    costs.append(compute_cost(request_graphs[request_index]))
                if utils:
                    if 'GPU' in physical.nodes[0]:
                        cpu_util, link_util, gpu_util, mem_util = compute_utils(
                            physical)
                        cpu_utils.append(cpu_util)
                        link_utils.append(link_util)
                        gpu_utils.append(gpu_util)
                        memory_utils.append(mem_util)
                    cpu_util, link_util = compute_utils(physical)
                    cpu_utils.append(cpu_util)
                    link_utils.append(link_util)

            if verbose:
                if request_embedded:
                    print(
                        f'\033[92m request {request_index} embedded successfully')
                else:
                    print(
                        f'\033[93m request {request_index} could not embedded')
            request_index += 1

        probs.append(blockeds/num)

        new_embeddeds = []
        for e in embeddeds:
            embedded_time = e[0]
            request = e[1]
            if t - embedded_time == request.graph['LifeTime']:
                free_embedded_request(physical, request)
                if verbose:
                    print(f'\033[94m free embedded graph {e[2]} successfully')
            else:
                new_embeddeds.append(e)

        embeddeds = new_embeddeds

    return_values = [probs, blockeds, num]
    if cost_revenue:
        return_values.append(costs)
        return_values.append(revenues)
    if utils:
        return_values.append(cpu_utils)
        return_values.append(link_utils)
        if 'GPU' in physical.nodes[0]:
            return_values.append(gpu_utils)
            return_values.append(memory_utils)
    return return_values
