
import networkx as nx
import numpy as np


def sample_from_physical(physical_graph, pred, from_cluster):
    r"""
      return a node from one cluster
    """
    cluster_index = np.where(pred == from_cluster)[0]
    sample_node = np.random.choice(cluster_index, 1, replace=False)[0]
    return sample_node


def get_graph_which_stisfy_bandwidth(graph, src, dst, min_band_width):
    adj = nx.to_numpy_matrix(graph, weight='Bandwidth')
    remove_lower_edges = np.vectorize(
        lambda x: x if x >= min_band_width else 0)
    return nx.from_numpy_matrix(remove_lower_edges(adj))


def has_shortest_path(graph, src, dst, min_band_width):
    g = get_graph_which_stisfy_bandwidth(graph, src, dst, min_band_width)
    return nx.has_path(g, src, dst)


def get_shortest_path(graph, src, dst, min_band_width):
    g = get_graph_which_stisfy_bandwidth(graph, src, dst, min_band_width)
    return nx.shortest_path(g, src, dst)


def can_embed(node_number, request_node_number, request_graph, physical_graph, verbose=False):
    r"""
    At first we check if node had resources

    Then we check if there were a path from request node to other adjacents 
    embeddings in physical graph.

    Returns: can embed a node based on other adjacent node embedding
    """
    node_has_resource = physical_graph.nodes[node_number][
        'CPU'] >= request_graph.nodes[request_node_number]['CPU']
    if verbose:
        print(
            f'Node {node_number} has resource for {request_node_number}: {node_has_resource}')

    if not node_has_resource:
        return False

    adjacents_bandwidth_satisfied = True
    for edge in request_graph.edges(request_node_number):
        src = edge[0]
        dst = edge[1]
        if 'Embedded' in request_graph.nodes[dst] and request_graph.nodes[dst]['Embedded'] != node_number:
            embedded_node = request_graph.nodes[dst]['Embedded']

            if not has_shortest_path(physical_graph, node_number, embedded_node, request_graph.edges[src, dst]['Bandwidth']):
                adjacents_bandwidth_satisfied = False
                if verbose:
                    print(
                        f'Can not embedd because of edge({src}, {dst}) in request graph can not embedded on ({node_number} ,..., {embedded_node}) path')
                break

    return adjacents_bandwidth_satisfied



def embed_to_physical(node_number, request_node_number, request_graph, physical_graph, verbose=False):
    r"""
    Embed a single node to physical graph
    """
    req_dict = request_graph.nodes(data=True)[request_node_number]
    request_graph.nodes[request_node_number]['Embedded'] = node_number
    physical_graph.nodes[node_number]['CPU'] -= req_dict['CPU']
    if verbose:
        print(
            f'Resources had changed for node {node_number} in physical graph.')
    for edge in request_graph.edges(request_node_number):
        src = edge[0]
        dst = edge[1]
        if 'Embedded' in request_graph.nodes[dst] and request_graph.nodes[dst]['Embedded'] != node_number:
            embedded_node = request_graph.nodes[dst]['Embedded']

            shortest_path = get_shortest_path(
                physical_graph, node_number, embedded_node, request_graph.edges[src, dst]['Bandwidth'])
            request_graph.edges[src, dst]['Embedded'] = []
            for i in range(len(shortest_path) - 1):
                p_src = shortest_path[i]
                p_dst = shortest_path[i+1]
                physical_graph.edges[p_src,
                                     p_dst]['Bandwidth'] -= request_graph.edges[src, dst]['Bandwidth']
                request_graph.edges[src, dst]['Embedded'] += [(p_src, p_dst)]

                if verbose:
                    print(
                        f'Bandwidth reduced between nodes {p_src}, {p_dst} in physical graph')

    request_graph.graph['Embedded'] = True



def free_embedded_request_node(physical_graph, request_graph, request_node):
    if 'Embedded' in request_graph.nodes[request_node]:
        embedded_node = request_graph.nodes[request_node]['Embedded']
        physical_graph.nodes[embedded_node]['CPU'] += request_graph.nodes[request_node]['CPU']
        del request_graph.nodes[request_node]['Embedded']
    for i in request_graph.edges(request_node):  # free links
        src = i[0]
        dst = i[1]
        if 'Embedded' in request_graph.edges[src, dst]:
            embeddeds = request_graph.edges[src, dst]['Embedded']
            for e in embeddeds:
                physical_graph.edges[e[0], e[1]
                                     ]['Bandwidth'] += request_graph.edges[src, dst]['Bandwidth']
            del request_graph.edges[src, dst]['Embedded']



def embed_request_in_cluster(request_graph, physical_graph, center_node, max_visit, max_depth=3, verbose=False, check_verbose=False, embedding_verbose=False):
    r"""
    Try to embed a request in some adjacent nodes of a center selected from a cluster 
    It uses BFS to traverse graph and in each itteration it selected best nodes in a depth

    Returns: embedding was successfull or not
    """
    q = request_graph
    sorted_request_nodes = sorted(
        q.nodes(data=True), key=lambda t: t[1]['CPU'], reverse=True)  # descending
    max_visit_in_every_depth = int(np.power(max_visit, 1/max_depth))

    for node in sorted_request_nodes:
        i = node[0]
        node_embedded = False
        depth = 0
        visited = physical_graph.number_of_nodes() * [False]
        queue = [(center_node, depth)]
        visited[center_node] = True

        while queue:
            (current_node, depth) = queue.pop(0)
            if depth > max_depth:
                break

            if can_embed(current_node, i, q, physical_graph, verbose=check_verbose):
                try:
                    embed_to_physical(current_node, i, q,
                                      physical_graph, verbose=embedding_verbose)
                    node_embedded = True
                    break
                except nx.NetworkXNoPath:  # In a rare case of having no path after embed some bandwidths
                    free_embedded_request_node(physical_graph, q, i)

            if depth == max_depth:
                continue

            sorted_edges = sorted(physical_graph.edges(
                center_node, data=True), key=lambda t: physical_graph.nodes[t[1]]['CPU'], reverse=True)  # desending
            sorted_edges = sorted_edges if len(
                sorted_edges) <= max_visit else sorted_edges[:max_visit_in_every_depth]

            for edge in sorted_edges:
                dst = edge[1]
                if not visited[dst]:
                    queue.append((dst, depth+1))
                    visited[dst] = True

        if verbose:
            if not node_embedded:
                print(f'can not embed node {i} in request.')
            else:
                print(f'Node {i} of request embedded.')

        if not node_embedded:
            return False
        elif node_embedded and i == q.number_of_nodes() - 1:
            return True




def free_embedded_request(physical_graph, request_graph):
    for i in request_graph.nodes:  # free cpus
        if 'Embedded' in request_graph.nodes[i]:
            embedded_node = request_graph.nodes[i]['Embedded']
            physical_graph.nodes[embedded_node]['CPU'] += request_graph.nodes[i]['CPU']
            del request_graph.nodes[i]['Embedded']

    for i in request_graph.edges:  # free links
        src = i[0]
        dst = i[1]
        if 'Embedded' in request_graph.edges[src, dst]:
            embeddeds = request_graph.edges[src, dst]['Embedded']
            for e in embeddeds:
                physical_graph.edges[e[0], e[1]
                                     ]['Bandwidth'] += request_graph.edges[src, dst]['Bandwidth']
            del request_graph.edges[src, dst]['Embedded']


def unfree_embedded_request(physical_graph, request_graph):
    for i in request_graph.nodes:  # unfree cpus
        if 'Embedded' in request_graph.nodes[i]:
            embedded_node = request_graph.nodes[i]['Embedded']
            physical_graph.nodes[embedded_node]['CPU'] -= request_graph.nodes[i]['CPU']

    for i in request_graph.edges:  # unfree links
        src = i[0]
        dst = i[1]
        if 'Embedded' in request_graph.edges[src, dst]:
            embeddeds = request_graph.edges[src, dst]['Embedded']
            for e in embeddeds:
                physical_graph.edges[e[0], e[1]
                                     ]['Bandwidth'] -= request_graph.edges[src, dst]['Bandwidth']


def embed_request(cluster_center, physical, request_graph, beta=30, verbose=True):
    r"""
    Try to embed request in some adjacent nodes from different clusters.
    beta is coeffience of max_visit_nodes
    Returns: embedding was successfull or not
    """
    request_embedded = False

    max_visit_nodes = request_graph.number_of_nodes() * beta
    request_embedded = embed_request_in_cluster(
        request_graph, physical, cluster_center, max_visit_nodes, verbose=False)  # can embedd to them?

    if not request_embedded:
        free_embedded_request(physical, request_graph)
    if verbose:
        if not request_embedded:
            print(f'could not embed request.')
        else:
            print(f'Request embedded.')

    return request_embedded, physical, request_graph


from compare_utils import compute_cost, compute_revenue

def graphViNE_embed(physical_graph, cluster_index, request_graph, alpha = 0.5, N_CLUSTERS=4, verbose=False):
  embeddeds = []
  embedded = False
  for i in range(N_CLUSTERS):
    j = int( len(np.where(cluster_index == i)[0]) * alpha )

    request_embedded = False
    while not request_embedded and j != 0:
      selected_node = sample_from_physical(physical_graph, cluster_index, i)
      request_embedded, physical_graph, request_graph = embed_request(selected_node, physical_graph, request_graph, verbose=verbose)
      j -= 1
  
    if request_embedded:
      embeddeds.append((request_graph.copy(), compute_cost(request_graph)))
      free_embedded_request(physical_graph, request_graph)

  if len(embeddeds) != 0:
    (q, cost) = sorted(embeddeds, key=lambda t: t[1])[0]
    request_graph = q
    embedded = True
    unfree_embedded_request(physical_graph, request_graph)

  return embedded, physical_graph, request_graph

