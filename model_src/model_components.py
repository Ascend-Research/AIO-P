import torch
import numpy as np
from params import *
from utils.model_utils import  torch_geo_batch_to_data_list, device

try:
    from torch_geometric.data import Data, Batch
except ModuleNotFoundError:
    print("Did not find torch_geometric, GNNs will be unavailable")


class Identity(torch.nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x, *args):
        return x


class GraphAggregator(torch.nn.Module):

    def __init__(self, hidden_size, gnn_constructor, aggr_method="last",
                 gnn_activ=None):
        super(GraphAggregator, self).__init__()
        self.aggr_method = aggr_method
        self.gnn_aggr_layer = PreEmbeddedGraphEncoder(hidden_size, hidden_size, hidden_size,
                                                      gnn_constructor,
                                                      activ=gnn_activ, n_layers=1)

    @staticmethod
    def _get_gnn_aggr_tsr_list(batch_size, n_nodes):
        src_list = [i for i in range(n_nodes - 1)]
        dst_list = [n_nodes - 1] * (n_nodes - 1)
        return [torch.LongTensor([src_list, dst_list]) for _ in range(batch_size)]

    def forward(self, node_embedding, batch_last_node_idx_list, index=None):
        assert len(node_embedding.shape) == 3
        if self.aggr_method == "sum":
            graph_embedding = node_embedding.sum(dim=1)
        elif self.aggr_method == "last":
            graph_embedding = node_embedding[:, -1, :]
        elif self.aggr_method == "mean":
            graph_embedding = node_embedding.mean(dim=1)
        elif self.aggr_method == "gnn":
            aggr_edge_tsr_list = self._get_gnn_aggr_tsr_list(node_embedding.shape[0], node_embedding.shape[1])
            node_embedding = self.gnn_aggr_layer(node_embedding, aggr_edge_tsr_list, batch_last_node_idx_list)
            graph_embedding = node_embedding[:, -1, :]
        elif self.aggr_method == "indexed":
            index = index.to(device())
            index = index.reshape(-1, 1, 1).repeat(1, 1, node_embedding.size(-1))
            graph_embedding = torch.gather(node_embedding, 1, index).squeeze(1)
        elif self.aggr_method == "none":
            graph_embedding = node_embedding
        elif self.aggr_method == "squeeze":
            assert len(node_embedding.shape) == 3 and node_embedding.shape[1] == 1, \
                "Invalid input shape: {}".format(node_embedding.shape)
            graph_embedding = node_embedding.squeeze(1)
        elif self.aggr_method == "flat":
            graph_embedding = node_embedding.reshape(node_embedding.shape[0], -1)
        elif self.aggr_method == "de-batch":
            graph_embedding = node_embedding.reshape(-1, node_embedding.shape[-1])
        else:
            raise ValueError("Unknown aggr_method: {}".format(self.aggr_method))
        return graph_embedding


class PreEmbeddedGraphEncoder(torch.nn.Module):

    def __init__(self, in_channels, hidden_size, out_channels, gnn_constructor,
                 activ=torch.nn.Tanh(), n_layers=4, dropout_prob=0.0,
                 add_normal_prior=False, eigvec_size=0):
        super(PreEmbeddedGraphEncoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.gnn_layers = torch.nn.ModuleList()
        self.eigvec_size = eigvec_size
        self.gnn_constructor = gnn_constructor
        for i in range(n_layers):
            input_size, output_size = hidden_size, hidden_size
            if i == 0:
                input_size = in_channels
            if i == n_layers - 1:
                output_size = out_channels
            gnn_layer = gnn_constructor(input_size, output_size)
            self.gnn_layers.append(gnn_layer)
        self.activ = activ
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.init_ff = torch.nn.Linear(2 * in_channels, in_channels)
        if self.eigvec_size > 0:
            self.eig_ff = torch.nn.Linear(eigvec_size, in_channels)
        self.add_normal_prior = add_normal_prior

    def add_prior(self, embedding):
        prior = np.random.normal(size=embedding.shape)
        prior = torch.from_numpy(prior).float().to(device())
        embedding = torch.cat([embedding, prior], dim=-1)
        return self.init_ff(embedding)

    def forward(self, batch_node_tsr, edge_tsr_list, batch_last_node_idx_list, eigvecs=None):
        node_embedding = batch_node_tsr.to(device())

        if self.add_normal_prior and node_embedding.shape[1] == 1:
            node_embedding = self.add_prior(node_embedding)

        if eigvecs is not None and self.eigvec_size > 0:
            eigvecs = eigvecs.to(device())
            if self.training:
                flip_mat = torch.rand((eigvecs.shape[0], eigvecs.shape[1]))
                for i in range(flip_mat.shape[0]):
                    for j in range(flip_mat.shape[1]):
                        if flip_mat[i, j] > 0.75:
                            eigvecs[i, j, :] *= -1
            node_embedding = node_embedding + self.eig_ff(eigvecs[:, :, :self.eigvec_size])

        data_list = [Data(x=node_embedding[i, :], edge_index=edge_tsr_list[i].to(device()), edge_attr=None)
                     for i in range(node_embedding.shape[0])]
        torch_geo_batch = Batch.from_data_list(data_list)
        edge_index_tsr = torch_geo_batch.edge_index
        curr_gnn_output = torch_geo_batch.x
        for li, gnn_layer in enumerate(self.gnn_layers):
            curr_gnn_output = gnn_layer(curr_gnn_output, edge_index_tsr)
            if self.activ is not None:
                curr_gnn_output = self.activ(curr_gnn_output)
        curr_gnn_output = self.dropout(curr_gnn_output)
        batch_embedding_list = torch_geo_batch_to_data_list(curr_gnn_output, batch_last_node_idx_list,
                                                            batch_indicator=torch_geo_batch.batch)
        batch_embedding = torch.cat([t.unsqueeze(0) for t in batch_embedding_list], dim=0)
        return batch_embedding
