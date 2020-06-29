
import torch
from sklearn.preprocessing import StandardScaler
import torch_geometric
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


# * Node number of the substrate network: **100**
# * Connection probability of the substrate network: **0.4 (Uniform Dist.)**
# * Initial available computing resources on substrate nodes: **0-100**
# * Initial available bandwidth resources on substrate links Average: **0-100**
# * lifetime of the VNRs: **100-900 time units (Normal Dist.)**
# * Bandwidth demand of a virtual link: **0-10**
# * Computing resource demand of a virtual node
# * Node number in a VNR: **3-10**
# * Connection probability of the virtual network: **0.7 (Uniform Dist.)**
#
#

# This method just use CPU as node feature. Instead of CPU, Hard, and Memory in the main implementation.
def create_network_graph(nodes_num=100, max_feature_val=400, min_feature_val=100, connection_prob=0.4, life_time=None, max_link_val=400, min_link_val=100):
    G = nx.Graph()
    if life_time != None:
        G.graph['LifeTime'] = np.random.randint(life_time[0], life_time[1])
    random_features = np.random.randint(
        min_feature_val, max_feature_val+1, size=(nodes_num, 1))
    for i, node_feature in enumerate(random_features):
        G.add_node(i, CPU=node_feature[0], MaxCPU=node_feature[0])

    for i in range(nodes_num):
        for j in range(i+1, nodes_num):
            uniform_sample = np.random.uniform(0, 1)
            if uniform_sample < connection_prob:
                bandwidth = np.random.randint(min_link_val, max_link_val+1)
                G.add_edge(i, j, Bandwidth=bandwidth, MaxBandwidth=bandwidth)

    for i in range(nodes_num):
        edges = list(G.edges(i))
        degree = len(edges)
        mean_bandwidth = 0
        for e in edges:
            mean_bandwidth += G.get_edge_data(*e)['Bandwidth']
        mean_bandwidth = mean_bandwidth/degree if degree != 0 else 0
        # G.nodes[i]['Degree'] = degree # In case if needed feature engineering
        # G.nodes[i]['MeanBandwidth'] = mean_bandwidth
        # nx.random_graphs.gnp_random_graph()
    return G


def draw_graph(G, width=0.05, pred=None):
    pos = nx.spring_layout(G)
    size = 20
    if pred != None:
        size = [G.nodes[i]['CPU'] for i in range(G.number_of_nodes())]
    options = {
        "node_color": pred if pred else 'blue',
        "node_size": size,
        "line_color": "grey",
        "linewidths": 0,
        "width": width,
        "cmap": plt.cm.brg,
    }

    nx.draw(G, **options)
    plt.show()


def normailize_data(l, oneD=False):
    npl = np.array(l).reshape(-1, 1) if oneD else np.array(l)
    std = StandardScaler().fit_transform(npl).astype('float32')
    std = list(std.reshape(-1)) if oneD else list(std)
    return std


def from_networkx(G, normalize=False):
    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        x = []
        for key, value in feat_dict.items():
            if key == 'MaxCPU':
                continue
            x.append(float(value))
        data['x'] = [x] if i == 0 else data['x'] + [x]
    if normalize:
        data['x'] = normailize_data(data['x'])

    # for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
    #   edge_attrs = []
    #   for key, value in feat_dict.items():
    #     if key == 'MaxBandwidth':
    #       continue
    #     edge_attrs.append(float(value))
    #   data['edge_attr'] = edge_attrs if i == 0 else data['edge_attr'] + edge_attrs

    edge_features = np.array(list(G.edges(data=True)))
    get_bandwidth = np.vectorize(lambda i: np.float32(i['Bandwidth']))
    data['edge_attr'] = list(get_bandwidth(edge_features[:, 2]))
    # if normalize: data['edge_attr'] = normailize_data(data['edge_attr'], oneD=True)

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data
