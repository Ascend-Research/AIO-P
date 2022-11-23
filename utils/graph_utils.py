import numpy as np
import collections


def hash_module(matrix, labeling):
  """
  Computes a graph-invariance MD5 hash of the matrix and label pair.

  Args:
    matrix: np.ndarray square upper-triangular adjacency matrix.
    labeling: list of int labels of length equal to both dimensions of
      matrix.

  Returns:
    MD5 hash of the matrix and labeling.
  """
  import hashlib
  vertices = np.shape(matrix)[0]
  in_edges = np.sum(matrix, axis=0).tolist()
  out_edges = np.sum(matrix, axis=1).tolist()

  assert len(in_edges) == len(out_edges) == len(labeling)
  hashes = list(zip(out_edges, in_edges, labeling))
  hashes = [hashlib.md5(str(h).encode('utf-8')).hexdigest() for h in hashes]
  # Computing this up to the diameter is probably sufficient but since the
  # operation is fast, it is okay to repeat more times.
  for _ in range(vertices):
    new_hashes = []
    for v in range(vertices):
      in_neighbors = [hashes[w] for w in range(vertices) if matrix[w, v]]
      out_neighbors = [hashes[w] for w in range(vertices) if matrix[v, w]]
      new_hashes.append(hashlib.md5(
          (''.join(sorted(in_neighbors)) + '|' +
           ''.join(sorted(out_neighbors)) + '|' +
           hashes[v]).encode('utf-8')).hexdigest())
    hashes = new_hashes
  fingerprint = hashlib.md5(str(sorted(hashes)).encode('utf-8')).hexdigest()

  return fingerprint


def edge_pairs_to_edge_list(edge_pairs):
    return [[p[0] for p in edge_pairs], [p[1] for p in edge_pairs]]


def edge_list_to_edge_pairs(edge_list):
    rv = list(zip(edge_list[0], edge_list[1]))
    return sorted(rv)


def adj_dict_to_edge_list(adj_dict):
    edge_pairs = []
    for src, dst_inds in adj_dict.items():
        for dst in dst_inds:
            edge_pairs.append((src, dst))
    return edge_pairs_to_edge_list(edge_pairs)


def edge_list_to_edge_matrix(edge_list, n_nodes):
    # Slightly different input params compared with edge_list_to_adj_mat(...)
    matrix = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for idx in range(len(edge_list[0])):
        matrix[edge_list[0][idx]][edge_list[1][idx]] = 1
    return matrix

def topo_sort_dfs(nodes, adj_dict):
    visited = set()
    results = []
    for node in nodes:
        if node not in visited:
            _topo_sort_dfs(node, adj_dict, visited, results)
    return results


def _topo_sort_dfs(node, adj_dict, visited, results):
    chds = adj_dict[node]
    visited.add(node)
    for chd in chds:
        if chd not in visited:
            _topo_sort_dfs(chd, adj_dict, visited, results)
    results.append(node)


def get_reverse_adj_dict(src2dsts, allow_self_edges=False):
    dst2srcs = collections.defaultdict(set)
    for src, dsts in src2dsts.items():
        for dst in dsts:
            if not allow_self_edges:
                assert src != dst, "src: {}, dst: {}".format(src, dst)
            dst2srcs[dst].add(src)
    return dst2srcs


def get_index_based_input_inds(node_ids, src2dsts):
    # Input node_ids must be topologically sorted
    dst2srcs = get_reverse_adj_dict(src2dsts, allow_self_edges=False)
    node_id2idx = {nid: ni for ni, nid in enumerate(node_ids)}
    graph_input_inds = []
    for ni, node_id in enumerate(node_ids):
        input_ids = dst2srcs[node_id]
        node_input_inds = [node_id2idx[_id] for _id in input_ids]
        node_input_inds.sort()
        assert all(i < ni for i in node_input_inds), "{}, {}".format(ni, node_input_inds)
        graph_input_inds.append(node_input_inds)
    return graph_input_inds


