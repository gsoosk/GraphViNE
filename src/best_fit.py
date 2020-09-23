import networkx as nx
import numpy as np
import sys
import traceback
from first_fit import Net, Inst

class BestFit:
    def __init__(self):
        pass

    def best_fit(self, physical, request):
        r"""
        vrn_inst: consists of v_net , p_net
        net.cpu: 1d array of each node cpu
        net.bw: adj matrix of net with value of bandwidths
        """
        vnr_inst = Inst(physical, request)
        return self.solve(vnr_inst)

    def solve(self, vnr_inst):
        best_vector_p = vnr_inst.p_net.cpu
        if vnr_inst.extra_features:
            best_vector_p = vnr_inst.p_net.gpu
            best_vector_p = vnr_inst.p_net.cpu
        best_vector_v = vnr_inst.v_net.cpu
        if vnr_inst.extra_features:
            best_vector_p = vnr_inst.p_net.gpu
            best_vector_p = vnr_inst.p_net.cpu
        (blocked, Fn, cp_residual) = self.greedy_node_mapping(
            vnr_inst, best_vector_p, best_vector_v)
        if blocked == False:
            rejected, Fl = self.shortest_path_link_mapping(vnr_inst, Fn)
        else:
            Fl = {}
            rejected = True
        return (rejected, Fn, Fl)

    def shortest_path_link_mapping(self, vnr_inst, Fn):
        v_len = len(vnr_inst.v_net.cpu)
        p_len = len(vnr_inst.p_net.cpu)
        rejected = False
        Fl = {}
        for i in range(v_len):
            for j in range(v_len):
                if vnr_inst.v_net.bw[i, j] != 0:
                    lp_temp = np.array(vnr_inst.p_net.bw)
                    for t in range(p_len):
                        for z in range(t, p_len):
                            if lp_temp[t, z] == 0 or lp_temp[t, z] < vnr_inst.v_net.bw[i, j]:
                                lp_temp[t, z] = 0
                                lp_temp[z, t] = 0
                            else:
                                lp_temp[t, z] = 1
                                lp_temp[z, t] = 1
                    i_physical = Fn[i]
                    j_physical = Fn[j]
                    try:
                        G = nx.Graph(lp_temp)
                    except:
                        traceback.print_exc()
                        rejected = True
                        return (rejected, Fl)
                    try:
                        i_j_path = nx.shortest_path(G, i_physical, j_physical)
                        for y in range(len(i_j_path)-1):
                            src = int(i_j_path[y])
                            dst = int(i_j_path[y+1])
                            # print '(before) link reduction: ', i, j, src, dst, vnr_inst.p_net.bw[src,dst]
                            vnr_inst.p_net.bw[src,
                                              dst] -= vnr_inst.v_net.bw[i, j]
                            vnr_inst.p_net.bw[dst,
                                              src] -= vnr_inst.v_net.bw[i, j]
                            if (i, j) in Fl:
                                Fl[(i, j)] += [(src, dst)]
                            else:
                                Fl[(i, j)] = [(src, dst)]
                            # print '(after) link reduction: ', i, j, src, dst, vnr_inst.p_net.bw[src,dst]
                    except:
                        # exception for no path found
                        # traceback.print_exc()
                        rejected = True
                        return (rejected, Fl)
        return (rejected, Fl)

    def greedy_node_mapping(self, vnr_inst, grc_vector_p, grc_vector_v):
        p_len = len(vnr_inst.p_net.cpu)
        v_len = len(vnr_inst.v_net.cpu)
        Fn = np.zeros(v_len)
        count = 0
        blocked = False
        index_p = np.flip(np.argsort(grc_vector_p), axis=0)
        # index_v = np.flip(np.argsort(grc_vector_v), axis=0)
        selected = np.ones(p_len)
        for i in range(v_len):
            for j in index_p:
                '''if selected[j] == 0:
                    continue'''
                if vnr_inst.p_net.cpu[j] >= vnr_inst.v_net.cpu[i]:
                    if vnr_inst.extra_features:
                        if vnr_inst.p_net.gpu[j] < vnr_inst.v_net.gpu[i] or vnr_inst.p_net.memory[j] < vnr_inst.v_net.memory[i]:
                            continue
                    Fn[i] = j
                    count = count + 1
                    vnr_inst.p_net.cpu[j] -= vnr_inst.v_net.cpu[i]
                    if vnr_inst.extra_features:
                        vnr_inst.p_net.gpu[j] -= vnr_inst.v_net.gpu[i]
                        vnr_inst.p_net.memory[j] -= vnr_inst.v_net.memory[i]
                    #selected[j] = 0
                    break
        if count != v_len:
            blocked = True
        return (blocked, Fn, None)




def best_fit_embed(physical_graph, request_graph, verbose=False):
    bf = BestFit()
    (rejected, Fn, Fl) = bf.best_fit(physical_graph, request_graph)

    if rejected:
        return not rejected, physical_graph, request_graph

    for i in range(len(Fn)):
        request_graph.nodes[i]['Embedded'] = int(Fn[i])
        physical_graph.nodes[int(
            Fn[i])]['CPU'] -= request_graph.nodes[i]['CPU']

    for key in Fl.keys():
        (i, j) = key
        request_graph.edges[i, j]['Embedded'] = Fl[key]
        for edge in Fl[key]:
            physical_graph.edges[edge[0], edge[1]
                                 ]['Bandwidth'] -= request_graph.edges[i, j]['Bandwidth']

    return not rejected, physical_graph, request_graph
