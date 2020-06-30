from grc import GRC, Net, Inst
import networkx as nx
import numpy as np

class NeuroVine:
    def __init__(self, physical, request):
        self.inst = Inst(physical, request)
        self.kappa = 3
        #print 'neurovine kappa: %d' %self.kappa

    def hf_grc(self):
        a = self.solve()
        p_net = self.inst.p_net.get_copy()
        self.inst.p_net.shrink_net(a)
        grc = GRC()
        rejected, Fn, Fl = grc.solve(self.inst)
        if rejected:
            self.inst.p_net = p_net
        else:
            p_net.cpu[a] = self.inst.p_net.cpu[a]
            self.inst.p_net.cpu = p_net.cpu
        info = {'v_net': self.inst.v_net, 'Fn': Fn, 'Fl': Fl}
        return rejected, info


    def solve(self):
        chi = self.node_ranks()
        #print('chi shape', chi.shape)
        psi = self.edge_ranks()
        #print('psi shape', psi.shape)
        zeta = self.preselected_nodes()
        if zeta > len(self.inst.p_net.cpu):
            zeta = len(self.inst.p_net.cpu)
        (T, I) = self.hopfield_network(chi, psi, zeta)
        #np.random.seed(1)
        #I = np.random.uniform(0,1,5)
        #T = np.random.uniform(0,1,(5,5))
        #print('ti', I, T)
        V = self.execute_hopfield(T, I)
        a = []
        for i in range(len(V)):
            if V[i] > 0.5:
                a.append(i)
        return a

    def execute_hopfield(self, T, I):
        u0 = 1.0
        Delta = np.inf
        delta = 1.0
        iMax = 2000
        i = 0
        U = np.random.uniform(0, 1, I.shape)
        V = np.copy(U)
        #print('U shape', U.shape, 'V shape', V.shape)
        tau = 0.02
        while (Delta > delta) and (i < iMax):
            i = i + 1
            #print(i)
            k1 = np.matmul(T,V) + I - U
            #print('k1 shape', k1.shape)
            k2_temp = np.divide((U + 0.5*tau*k1), u0)
            #print('k2_temp shape', k2_temp.shape)
            k2 = np.matmul(T, (0.5 * (1+np.tanh(k2_temp)))) + I - (U+0.5*tau*k1)
            #print('k2 shape', k2.shape)
            k3_temp = np.divide((U-tau*k1+2*tau*k2), u0)
            #print('k3_temp_shape', k3_temp.shape)
            k3 = np.matmul(T, (0.5 * (1+np.tanh(k3_temp)))) + I - (U-tau*k1+2*tau*k2)
            #print('k3 shape', k3.shape)
            dU = (k1+4*k2+k3)/6
            #print('dU shape', dU.shape)
            U = U + tau*dU
            #print('u', U)
            # Delta = np.abs()
            Delta = np.sqrt(sum(np.square(dU)))
            V = 0.5 * (1 + np.tanh(np.divide(U, u0)))
            #print('v', V)
            #print('U shape', U.shape, 'V shape', V.shape)
            #break
        return V

    def hopfield_network(self, chi, psi, zeta):
        alpha = 10
        t_const = np.ones((len(chi), len(chi)))
        for i in range(zeta):
            t_const[i,i] = 0
        i_const = np.ones(len(chi)) * -(2*zeta-1)
        T = -2 * (psi + alpha * t_const)
        I = -(chi + alpha * i_const)
        return (T, I)

    def node_ranks(self):
        beta = 7.0
        cp_max = max(self.inst.p_net.cpu)
        chi = beta * (cp_max - self.inst.p_net.cpu) / cp_max
        #print('cp_max', cp_max)
        #print('cpu', self.inst.p_net.cpu)
        #print('chi', chi)
        return chi

    def edge_ranks(self):
        n = len(self.inst.p_net.cpu)
        gamma = 3.0
        b_max = np.amax(self.inst.p_net.bw)
        wHf = b_max - (np.divide(self.inst.p_net.bw, b_max))
        G = nx.Graph(wHf)
        d_t = nx.floyd_warshall(G)
        d_max = 0
        for i in range(n):
            for j in range(i, n):
                if d_t[i][j] > d_max:
                    d_max = d_t[i][j]
        '''if d_max == np.inf:
            print('d_max is inf')
        else:
            print('d_max is:', d_max)'''
        psi = np.ones((n,n))
        for i in range(n):
            for j in range(i, n):
                psi[i,j] = gamma * d_t[i][j] / d_max
                psi[j,i] = gamma * d_t[j][i] / d_max
        return psi

    def preselected_nodes(self):
        n = len(self.inst.v_net.cpu)
        #print('zeta', 5 * n)
        return n*self.kappa


def neuro_vine_embed(physical_graph, request_graph, verbose=False):
    neuro_vine = NeuroVine(physical_graph, request_graph)
    (rejected, info) = neuro_vine.hf_grc()
    Fn = info['Fn']
    Fl = info['Fl']

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