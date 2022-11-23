import torch
from utils.model_utils import device
from model_src.model_components import GraphAggregator

class CGNodeEmbedding(torch.nn.Module):

    def __init__(self, n_unique_labels, out_embed_size, shape_embed_size, kernel_embed_size,
                 bias_embed_size=2, n_unique_kernels=8, n_shape_vals=6):
        super(CGNodeEmbedding, self).__init__()
        assert kernel_embed_size % 2 == 0
        self.out_embed_size = out_embed_size
        self.n_unique_labels = n_unique_labels
        self.n_unique_kernels = n_unique_kernels
        self.n_shape_vals = n_shape_vals
        regular_out_embed_size = out_embed_size - shape_embed_size
        weighted_out_embed_size = out_embed_size - shape_embed_size - kernel_embed_size - bias_embed_size
        self.regular_embed_layer = torch.nn.Embedding(n_unique_labels, regular_out_embed_size)
        self.weighted_embed_layer = torch.nn.Embedding(n_unique_labels, weighted_out_embed_size)
        self.kernel_embed_layer = torch.nn.Embedding(n_unique_kernels, kernel_embed_size // 2)
        self.shape_embed_layer = torch.nn.Linear(n_shape_vals, shape_embed_size)
        self.bias_embed_layer = torch.nn.Linear(2, bias_embed_size)

    def forward(self, regular_node_inds, regular_node_shapes,
                weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias):
        regular_embedding = None
        weighted_embedding = None
        if regular_node_inds is not None:
            regular_embedding = self.regular_embed_layer(regular_node_inds.to(device()))
            shape_embedding = self.shape_embed_layer(regular_node_shapes.to(device()))
            regular_embedding = torch.cat([regular_embedding,
                                           shape_embedding], dim=-1)
        if weighted_node_inds is not None:
            weighted_embedding = self.weighted_embed_layer(weighted_node_inds.to(device()))
            kernel_embedding = self.kernel_embed_layer(weighted_node_kernels.to(device()))
            kernel_embedding = kernel_embedding.view(kernel_embedding.shape[0],
                                                     kernel_embedding.shape[1], -1)
            shape_embedding = self.shape_embed_layer(weighted_node_shapes.to(device()))
            bias_embedding = self.bias_embed_layer(weighted_node_bias.to(device()))
            weighted_embedding = torch.cat([weighted_embedding,
                                            shape_embedding,
                                            kernel_embedding,
                                            bias_embedding], dim=-1)
        if regular_embedding is not None and weighted_embedding is not None:
            # Implicit rule of weighted node always in front, very important
            node_embedding = torch.cat([weighted_embedding, regular_embedding], dim=1)
        elif regular_embedding is not None:
            node_embedding = regular_embedding
        elif weighted_embedding is not None:
            node_embedding = weighted_embedding
        else:
            raise ValueError("Input to CGNodeEmbedding cannot both be None")
        return node_embedding

class CGRegressor(torch.nn.Module):

    def __init__(self, embed_layer, encoder, aggregator, hidden_size,
                 activ=None, ext_feat_size=0):
        super(CGRegressor, self).__init__()
        self.embed_layer = embed_layer
        self.encoder = encoder
        self.aggregator = aggregator
        self.activ = activ
        self.ext_feat_size = ext_feat_size
        self.hidden_size = hidden_size
        if self.ext_feat_size > 0:
            self.eigvalue_ff = torch.nn.Linear(ext_feat_size, hidden_size)
        self.post_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.regressor = torch.nn.Linear(hidden_size, 1)
        self.alpha, self.b, self.I = None, None, None

    def init_alpha(self):
        self.alpha = torch.nn.Parameter(torch.randn([1]))
        self.b = torch.nn.Parameter(torch.randn([1, self.hidden_size + 1]))
        self.I = torch.nn.Parameter(torch.ones([1, self.hidden_size + 1]), requires_grad=False)

    def forward(self, regular_node_inds, regular_node_shapes,
                weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                edge_tsr_list, batch_last_node_idx_list, index=None,
                ext_feat=None):

        if type(regular_node_inds) is list:
            regular_node_inds = regular_node_inds[0]
            regular_node_shapes = regular_node_shapes[0]
            weighted_node_inds = weighted_node_inds[0]
            weighted_node_shapes = weighted_node_shapes[0]
            weighted_node_kernels = weighted_node_kernels[0]
            weighted_node_bias = weighted_node_bias[0]
            edge_tsr_list = edge_tsr_list[0]
            batch_last_node_idx_list = batch_last_node_idx_list[0]

        node_embedding = self.embed_layer(regular_node_inds, regular_node_shapes,
                                          weighted_node_inds, weighted_node_shapes,
                                          weighted_node_kernels, weighted_node_bias)
        batch_embedding = self.encoder(node_embedding, edge_tsr_list, batch_last_node_idx_list, eigvecs=ext_feat[0])
        graph_embedding = self.aggregator(batch_embedding, batch_last_node_idx_list, index=index)

        if self.ext_feat_size > 0:
            eigvals = ext_feat[1].to(device())
            # Randomly flip eigenvalues during training
            if self.training:
                flip_mat = torch.rand((eigvals.shape[0]))
                for i in range(flip_mat.shape[0]):
                    if flip_mat[i] > 0.75:
                        eigvals[i, :] *= -1
            graph_embedding = graph_embedding + self.eigvalue_ff(eigvals[:, :self.ext_feat_size])

        graph_embedding = self.post_proj(graph_embedding)
        if self.activ is not None:
            graph_embedding = self.activ(graph_embedding)

        if self.alpha is not None:
            alpha_sparse = (self.alpha * self.I) + self.b
            rescale_w = torch.t(alpha_sparse[:, :-1] * self.regressor.weight)
            rescale_bias = alpha_sparse[:, -1] * self.regressor.bias
            out = torch.matmul(graph_embedding, rescale_w) + rescale_bias
        else:
            out = self.regressor(graph_embedding)
        return out


# This is just a granularized CGRegressor that just contains the Graph Embedding components.
class CGraphEmbedding(torch.nn.Module):

    def __init__(self, embed_layer, encoder, aggregator):
        super(CGraphEmbedding, self).__init__()
        self.embed_layer = embed_layer
        self.encoder = encoder
        self.aggregator = aggregator

    def forward(self, regular_node_inds, regular_node_shapes,
                weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                edge_tsr_list, batch_last_node_idx_list, index=None, ext_feat=None):

        node_embedding = self.embed_layer(regular_node_inds, regular_node_shapes,
                                          weighted_node_inds, weighted_node_shapes,
                                          weighted_node_kernels, weighted_node_bias)
        batch_embedding = self.encoder(node_embedding, edge_tsr_list, batch_last_node_idx_list, eigvecs=ext_feat)
        graph_embedding = self.aggregator(batch_embedding, batch_last_node_idx_list, index=index)

        return graph_embedding


def make_cg_regressor(n_unique_labels, out_embed_size,
                      shape_embed_size, kernel_embed_size,
                      n_unique_kernels, n_shape_vals,
                      hidden_size, out_channels, gnn_constructor,
                      bias_embed_size=2, gnn_activ=torch.nn.ReLU(),
                      n_gnn_layers=4, dropout_prob=0.0,
                      regressor_activ=None, aggr_method="mean",
                      eigvec_size=0, eigval_size=0):
    from model_src.model_components import PreEmbeddedGraphEncoder
    embed_layer = CGNodeEmbedding(n_unique_labels, out_embed_size, shape_embed_size, kernel_embed_size,
                                  bias_embed_size, n_unique_kernels, n_shape_vals)
    encoder = PreEmbeddedGraphEncoder(out_embed_size, hidden_size, out_channels, gnn_constructor,
                                      gnn_activ, n_gnn_layers, dropout_prob, eigvec_size=eigvec_size)
    aggregator = GraphAggregator(hidden_size, gnn_constructor, aggr_method=aggr_method)
    regressor = CGRegressor(embed_layer, encoder, aggregator, hidden_size, activ=regressor_activ,
                            ext_feat_size=eigval_size)
    return regressor


def make_fuzzy_cg_encoder(n_unique_labels, out_embed_size,
                          shape_embed_size, kernel_embed_size,
                          n_unique_kernels, n_shape_vals,
                          hidden_size, out_channels, gnn_constructor,
                          bias_embed_size=2, gnn_activ=torch.nn.ReLU(),
                          n_gnn_layers=4, dropout_prob=0.0, aggr_method="mean",
                          eigvec_size=0):
    from model_src.model_components import PreEmbeddedGraphEncoder
    embed_layer = CGNodeEmbedding(n_unique_labels, out_embed_size, shape_embed_size, kernel_embed_size,
                                  bias_embed_size, n_unique_kernels, n_shape_vals)
    encoder = PreEmbeddedGraphEncoder(out_embed_size, hidden_size, out_channels, gnn_constructor,
                                      gnn_activ, n_gnn_layers, dropout_prob, eigvec_size=eigvec_size)
    aggregator = GraphAggregator(hidden_size, gnn_constructor, aggr_method=aggr_method)

    return CGraphEmbedding(embed_layer, encoder, aggregator)

