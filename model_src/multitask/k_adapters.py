import torch
from utils.model_utils import device
import torch_geometric
from torch_geometric.data import Data, Batch
from utils.model_utils import torch_geo_batch_to_data_list
from model_src.model_components import Identity


class CGRegressorAdapter(torch.nn.Module):
    def __init__(self, base_model, K=1, ft_adapter=0):
        super(CGRegressorAdapter, self).__init__()
        self.base_embed_l = base_model.embed_layer
        self.base_encoder = base_model.encoder
        self.base_aggregator = base_model.aggregator
        self.base_post_proj = base_model.post_proj
        self.base_regressor = base_model.regressor
        if base_model.activ is not None:
            self.activ = base_model.activ
        else:
            self.activ = Identity
        self.K, self.k_curr = K, 0
        self.ft_adapter = ft_adapter
        self.hidden_size = base_model.hidden_size

        self.adapter_bodies = torch.nn.ModuleList()
        self.adapter_heads = torch.nn.ModuleList()
        self.alpha = None
        self.b = None 
        self.I_list = None

        for _ in range(K):
            self.adapter_bodies.append(PreEmbedGraphEncoderAdapter(base_model.encoder))
            self.adapter_heads.append(self._make_head())

    def init_alpha(self):
        self.alpha = torch.nn.Parameter(torch.randn([self.K]))
        self.b = torch.nn.Parameter(torch.randn([self.K, self.hidden_size + 1]))
        self.I = torch.nn.Parameter(torch.ones([1, self.hidden_size + 1]), requires_grad=False)

    def forward_rescale(self, graph_embedding, k):
        final_layer = self.adapter_heads[k][-1]
        graph_embedding = self.adapter_heads[k][:-1](graph_embedding)
        alpha_sparse = (self.alpha[k] * self.I) + self.b[k, :]
        rescale_w = torch.t(alpha_sparse[:, :-1] * final_layer.weight)
        rescale_bias = alpha_sparse[:, -1] * final_layer.bias
        return torch.matmul(graph_embedding, rescale_w) + rescale_bias


    def _make_head(self, multiplier=2):
        hs = self.hidden_size
        return torch.nn.Sequential(
            torch.nn.Linear(multiplier * hs, hs),
            torch.nn.Linear(hs, hs),
            torch.nn.ReLU(),
            torch.nn.Linear(hs, hs),
            torch.nn.Linear(hs, hs),
            self.activ(),
            torch.nn.Linear(hs, 1)
        )

    def set_k(self, k):
        if self.K == 1:
            self.k_curr = 0
        else:
            self.k_curr = min(k, self.K - 1)

    def forward(self, regular_node_inds, regular_node_shapes,
                weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                edge_tsr_list, batch_last_node_idx_list, index=None, ext_feat=None):

        if type(regular_node_inds) is list:
            regular_node_inds = regular_node_inds[0]
            regular_node_shapes = regular_node_shapes[0]
            weighted_node_inds = weighted_node_inds[0]
            weighted_node_shapes = weighted_node_shapes[0]
            weighted_node_kernels = weighted_node_kernels[0]
            weighted_node_bias = weighted_node_bias[0]
            edge_tsr_list = edge_tsr_list[0]
            batch_last_node_idx_list = batch_last_node_idx_list[0]

        node_embedding = self.base_embed_l(regular_node_inds, regular_node_shapes,
                                           weighted_node_inds, weighted_node_shapes,
                                           weighted_node_kernels, weighted_node_bias)

        if self.k_curr >= 0 or self.K == 1:
            graph_embedding = self.fwd_single_gnn(node_embedding, edge_tsr_list,
                                                  batch_last_node_idx_list, index)
            if self.alpha is not None:
                return self.forward_rescale(graph_embedding, self.k_curr)
            else:
                return self.adapter_heads[self.k_curr](graph_embedding)

        else:
            base_batch_embed, base_latent_embeds, edge_index_tsr, batch_ind = self.adapter_bodies[0].\
                compute_base_latent_embeds(node_embedding, edge_tsr_list, batch_last_node_idx_list)
            graph_embeddings = [self.base_aggregator(base_batch_embed, batch_last_node_idx_list, index=index)]

            latent_embeds = []
            for k in range(self.K):
                k_adapt_batch_embed = self.adapter_bodies[k].compute_adapter_embeds(base_latent_embeds, batch_ind,
                                                                                    edge_index_tsr,
                                                                                    batch_last_node_idx_list)
                graph_embeddings.append(self.base_aggregator(k_adapt_batch_embed, batch_last_node_idx_list,
                                                             index=index))
                pred_input = torch.cat([graph_embeddings[0], graph_embeddings[-1]], dim=-1)

                if self.alpha is not None:
                    k_prediction = self.forward_rescale(pred_input, k)
                else:
                    k_prediction = self.adapter_heads[k](pred_input)

                latent_embeds.append(k_prediction)

            concat_latent_embeds = torch.cat(latent_embeds, dim=-1)
            return torch.mean(concat_latent_embeds, dim=-1).unsqueeze(-1)

    def fwd_single_gnn(self, node_embedding, edge_tsr_list, batch_last_node_idx_list, index):
        batch_embed_base, batch_embed_adapt = self.adapter_bodies[self.k_curr](node_embedding, edge_tsr_list,
                                                                      batch_last_node_idx_list)
        graph_embed_base = self.base_aggregator(batch_embed_base, batch_last_node_idx_list, index=index)
        graph_embed_adapt = self.base_aggregator(batch_embed_adapt, batch_last_node_idx_list, index=index)
        graph_embedding = torch.cat([graph_embed_base, graph_embed_adapt], dim=-1)
        return graph_embedding

    def ft_parameters(self):

        params = []

        for k in range(self.K):
            params += list(self.adapter_heads[k].parameters())

        if self.ft_adapter > 0:
            if self.ft_adapter < 3:
                for param in self.base_encoder.gnn_layers.parameters():
                    param.requires_grad = True
                params += list(self.base_encoder.gnn_layers.parameters())
            if self.ft_adapter > 1:
                for k in range(self.K):
                    params += list(self.adapter_bodies[k].gnn_layers.parameters())
            else:
                for k in range(self.K):
                    for param in self.adapter_bodies[k].gnn_layers.parameters():
                        param.requires_grad = False
        return params


class PreEmbedGraphEncoderAdapter(torch.nn.Module):
    def __init__(self, base_encoder):
        super(PreEmbedGraphEncoderAdapter, self).__init__()
        self.base_encoder = base_encoder
        self.in_channels = base_encoder.in_channels
        self.hidden_size = base_encoder.hidden_size
        self.out_channels = base_encoder.out_channels
        self.gnn_layers = torch.nn.ModuleList()
        self.gnn_constructor = base_encoder.gnn_constructor

        for i in range(len(base_encoder.gnn_layers)):
            input_size, output_size = self.hidden_size, self.hidden_size
            if i == 0:
                input_size = self.in_channels
            if i == len(base_encoder.gnn_layers) - 1:
                output_size = self.out_channels
            gnn_layer = self.gnn_constructor(input_size * 2, output_size)
            self.gnn_layers.append(gnn_layer)
        self.activ = base_encoder.activ

    def forward(self, batch_node_tsr, edge_tsr_list, batch_last_node_idx_list):
        base_batch_embed, base_latent_embeds, edge_index_tsr, batch_ind = self.compute_base_latent_embeds(batch_node_tsr, edge_tsr_list, 
                                                                                          batch_last_node_idx_list)
        adapter_batch_embed = self.compute_adapter_embeds(base_latent_embeds, batch_ind, edge_index_tsr,
                                                          batch_last_node_idx_list)

        return base_batch_embed, adapter_batch_embed

    def compute_base_latent_embeds(self, batch_node_tsr, edge_tsr_list, batch_last_node_idx_list):
        node_embedding = batch_node_tsr.to(device())
        
        data_list = [Data(x=node_embedding[i, :], edge_index=edge_tsr_list[i].to(device()), edge_attr=None)
                    for i in range(node_embedding.shape[0])]
        torch_geo_batch = Batch.from_data_list(data_list)
        edge_index_tsr = torch_geo_batch.edge_index
        base_latent_embeds = [torch_geo_batch.x]

        for i in range(len(self.base_encoder.gnn_layers)):
            gnn_layer_output = self.base_encoder.gnn_layers[i](base_latent_embeds[-1], edge_index_tsr)
            if self.base_encoder.activ is not None:
                gnn_layer_output = self.base_encoder.activ(gnn_layer_output)
            base_latent_embeds.append(gnn_layer_output)
        
        base_batch_embed = torch_geo_batch_to_data_list(base_latent_embeds[-1], batch_last_node_idx_list,
                                                        batch_indicator=torch_geo_batch.batch)
        base_batch_embed = torch.cat([t.unsqueeze(0) for t in base_batch_embed], dim=0)

        return base_batch_embed, base_latent_embeds, edge_index_tsr, torch_geo_batch.batch

    def compute_adapter_embeds(self, base_latent_embeds, batch_ind, edge_tsr_list, batch_last_node_idx_list):
        curr_adapter_output = base_latent_embeds[0]
        for i in range(len(self.gnn_layers)):
            curr_adapter_output = torch.cat([base_latent_embeds[i + 1], curr_adapter_output], dim=-1)
            curr_adapter_output = self.gnn_layers[i](curr_adapter_output, edge_tsr_list)
            if self.activ is not None:
                curr_adapter_output = self.activ(curr_adapter_output)
        adapt_batch_embed = torch_geo_batch_to_data_list(curr_adapter_output, batch_last_node_idx_list,
                                                         batch_indicator=batch_ind)
        adapt_batch_embed = torch.cat([t.unsqueeze(0) for t in adapt_batch_embed], dim=0)
        return adapt_batch_embed


def load_kadapt_model_chkpt(chkpt, book_keeper, gnn_type="GraphConv", gnn_args="", in_channels=32, hidden_size=32, out_channels=32, 
                            num_layers=6, gnn_activ="tanh", reg_activ=None, aggr_method="mean"):

    from model_src.comp_graph.tf_comp_graph_models import make_cg_regressor
    from model_src.comp_graph.tf_comp_graph import OP2I
    from utils.model_utils import get_activ_by_name
    import re
    
    base_model = make_cg_regressor(n_unique_labels=len(OP2I().build_from_file()), out_embed_size=in_channels,
                              shape_embed_size=8, kernel_embed_size=8, n_unique_kernels=8, n_shape_vals=6,
                              hidden_size=hidden_size, out_channels=out_channels,
                              gnn_constructor=make_gnn_constructor(gnn_type, gnn_args=gnn_args),
                              gnn_activ=get_activ_by_name(gnn_activ), n_gnn_layers=num_layers,
                              dropout_prob=0., aggr_method=aggr_method,
                              regressor_activ=get_activ_by_name(reg_activ)).to(device())

    start_idx = re.search(r'_k..pt', chkpt).start() + 2
    end_idx = chkpt.index(".pt")
    num_k = int(chkpt[start_idx:end_idx])
    model = CGRegressorAdapter(base_model=base_model, K=num_k, ft_adapter=1)
    model.set_k(-1)
    if "_scale_ft_k" in chkpt or "_s_ft_k" in chkpt:
        model.init_alpha()
    
    book_keeper.load_model_checkpoint(model, allow_silent_fail=False, skip_eval_perfs=True, 
                                      checkpoint_file=chkpt)
    model.eval()
    model = model.to(device())
    return model

def make_gnn_constructor(gnn_type, gnn_args=None):
    if "GINConv" in gnn_type:
        def gnn_constructor(in_channels, out_channels):
            nn = torch.nn.Sequential(torch.nn.Linear(in_channels, in_channels),
                                     torch.nn.Linear(in_channels, out_channels),
                                     )
            return torch_geometric.nn.GINConv(nn=nn)
    elif "GraphTransformer" in gnn_type:
        def gnn_constructor(in_channels, out_channels):
            from model_src.model_components import GraphTransformerLayer
            return eval("GraphTransformerLayer(%d, %d, %s)"
                        % (in_channels, out_channels, gnn_args))
    else:
        def gnn_constructor(in_channels, out_channels):
            return eval("torch_geometric.nn.%s(%d, %d, %s)"
                        % (gnn_type, in_channels, out_channels, gnn_args))
    return gnn_constructor
