import networkx as nx
import numpy as np
import sys
import traceback


class Net:
    def __init__(self, cpu, bw):
        self.cpu = cpu
        self.bw = bw
    
    def get_copy(self): 
        return Net(self.cpu, self.bw)
    
    def shrink_net(self, a):
        cpu_temp_a = self.cpu[a]
        self.cpu[:] = 0
        self.cpu[a] = cpu_temp_a


class Inst:
    def __init__(self, physical, request):
        bw = np.asarray(nx.to_numpy_matrix(physical, weight='Bandwidth'))
        cpu = np.array(list(nx.get_node_attributes(physical, 'CPU').values()))
        self.p_net = Net(cpu, bw)
        bw = np.asarray(nx.to_numpy_matrix(request, weight='Bandwidth'))
        cpu = np.array(list(nx.get_node_attributes(request, 'CPU').values()))
        self.v_net = Net(cpu, bw)


class GRC:
    def __init__(self):
        pass

    def grc(self, physical, request):
        r"""
        vrn_inst: consists of v_net , p_net
        net.cpu: 1d array of each node cpu
        net.bw: adj matrix of net with value of bandwidths
        """
        vnr_inst = Inst(physical, request)
        return self.solve(vnr_inst)

    def solve(self, vnr_inst):
        grc_vector_p = self.grc_vector(vnr_inst.p_net, d=0.85)
        grc_vector_v = self.grc_vector(vnr_inst.v_net, d=0.85)
        (blocked, Fn, cp_residual) = self.greedy_node_mapping(
            vnr_inst, grc_vector_p, grc_vector_v)
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
        index_v = np.flip(np.argsort(grc_vector_v), axis=0)
        selected = np.ones(p_len)
        for i in index_v:
            for j in index_p:
                '''if selected[j] == 0:
                    continue'''
                if vnr_inst.p_net.cpu[j] >= vnr_inst.v_net.cpu[i]:
                    Fn[i] = j
                    count = count + 1
                    vnr_inst.p_net.cpu[j] -= vnr_inst.v_net.cpu[i]
                    #selected[j] = 0
                    break
        if count != v_len:
            blocked = True
        return (blocked, Fn, None)

    def grc_vector(self, net_inst, d):
        net_len = len(net_inst.cpu)
        M = np.array(net_inst.bw)
        for i in range(net_len):
            sum_adj_i = sum(net_inst.bw[:, i])
            sum_adj_v = np.ones(net_len) * sum_adj_i
            M[:, i] = np.divide(net_inst.bw[:, i], sum_adj_v)
        c = net_inst.cpu / sum(net_inst.cpu)
        r0 = c
        Delta = 9999999
        sigma = 0.00001
        while Delta >= sigma:
            r1 = (1-d) * c + d * np.matmul(M, r0)
            tmp = np.linalg.norm(r1 - r0, ord=2)
            Delta = tmp
            r0 = r1
        return r1


def grc_embed(physical_graph, request_graph, verbose=False):
    grc = GRC()
    (rejected, Fn, Fl) = grc.grc(physical_graph, request_graph)

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
