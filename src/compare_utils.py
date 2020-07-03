import networkx as nx
import numpy as np


def compute_revenue(req):
    revenue = 0
    for i in req.nodes:
        revenue += req.nodes[i]['CPU']
    for e in req.edges:
        revenue += req.edges[e[0], e[1]]['Bandwidth']
    return revenue


def compute_cost(req):
    cost = 0
    for i in req.nodes:
        cost += req.nodes[i]['CPU']

    for e in req.edges:
        num_of_occupied_edges = 0
        if 'Embedded' in req.edges[e[0], e[1]]:
            num_of_occupied_edges = len(req.edges[e[0], e[1]]['Embedded'])
        cost += req.edges[e[0], e[1]]['Bandwidth'] * num_of_occupied_edges
    return cost


def compute_utils(physical):
    cpu_util = 0
    link_util = 0

    for i in physical.nodes:
        cpu_util += 1 - (physical.nodes[i]['CPU']/physical.nodes[i]['MaxCPU'])
    cpu_util /= physical.number_of_nodes()

    for e in physical.edges:
        link_util += 1 - \
            (physical.edges[e[0], e[1]]['Bandwidth'] /
             physical.edges[e[0], e[1]]['MaxBandwidth'])
    link_util /= physical.number_of_edges()

    return cpu_util, link_util

