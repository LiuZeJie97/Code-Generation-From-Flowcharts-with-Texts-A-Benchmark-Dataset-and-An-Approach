"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv
from torch.nn.parameter import Parameter


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128, n_type = None, n_layer = None):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),

        )
        self.n_type = n_type
        self.n_layer = n_layer

    def forward(self, z, meta_path_list = None):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        # print(str(self.n_type) + " " +str( self.n_layer)  + str(self.project[0].weight))
        # beta_ = beta.detach().cpu()
        # if my_config.DEBUG == False:
        #     for i in range(len(beta)):
        #         wandb.log({
        #             "beta_" + self.n_type + "/" + str(self.n_layer) + "_" +str(meta_path_list[i]):beta_[i]
        #         })

        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, meta_paths, ntypes, in_size, out_size, layer_num_heads, dropout, n_layer = None):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            if meta_paths[i] == ["evidence2evidence"]:
                self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                               dropout, dropout, activation=F.elu,
                                               allow_zero_in_degree=True))
            else:
                self.gat_layers.append(GATConv((in_size,in_size), out_size, layer_num_heads,
                                               dropout, dropout, activation=F.elu,
                                               allow_zero_in_degree=True))
        self.semantic_attention_dic = nn.ModuleDict()
        # self.update_gate_claim = Parameter((torch.rand(out_size * (layer_num_heads), dtype=torch.float, requires_grad= True) * 0.2).to(my_config.device))
        # self.update_gate_evidence = Parameter((torch.rand(out_size * (layer_num_heads), dtype=torch.float, requires_grad=True) * 0.2).to(my_config.device))
        # self.update_gate_verb = Parameter((torch.rand(out_size * (layer_num_heads), dtype=torch.float, requires_grad=True) * 0.2 ).to(my_config.device))
        # self.update_gate_argument = Parameter((torch.rand(out_size * (layer_num_heads), dtype=torch.float, requires_grad=True) * 0.2).to(my_config.device))
        for ntype in ntypes:
            self.semantic_attention_dic[ntype] = SemanticAttention(in_size=out_size * (layer_num_heads), n_type = ntype, n_layer= n_layer)
            self.semantic_attention_dic[ntype].to(my_config.device)

            # self.semantic_attention = SemanticAttention(in_size=out_size * (layer_num_heads))
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        # self.update_gate = Parameter((torch.rand(out_size * (layer_num_heads), dtype=torch.float, requires_grad= True)*0.2).to(my_config.device))
        self.layer_num_heads = layer_num_heads
        self._cached_graph = None
        self._cached_coalesced_graph = {}

        self.in_size = in_size
        self.out_size = out_size
        self.layer_num_heads = layer_num_heads

        self.n_layer = n_layer
    def forward(self, g, features_dic):

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                new_g = dgl.metapath_reachable_graph(
                        g, meta_path)
                src_type = g.to_canonical_etype(meta_path[0])[0]
                dst_type = g.to_canonical_etype(meta_path[-1])[2]
                if src_type == dst_type:
                    new_g = new_g.add_self_loop()
                else:
                    number_of_src = new_g.number_of_nodes(src_type)
                    number_of_dst = new_g.number_of_nodes(dst_type)
                    new_g.add_nodes(number_of_dst, ntype=src_type)
                    src_list = [i + number_of_src for i in range(number_of_dst)]
                    dst_list = [i for i in range(number_of_dst)]
                    new_g.add_edges(src_list,dst_list)
                self._cached_coalesced_graph[meta_path] = new_g

        meta_path_list_dic = {}
        semantic_embeddings_dic = {}
        for ntype in g.ntypes:
            semantic_embeddings_dic[ntype] = []
            meta_path_list_dic[ntype] = []
            # meta_path_list_dic[ntype] = [(ntype + "_self_loop",)]

        for i, meta_path in enumerate(self.meta_paths):
            src_type = g.to_canonical_etype(meta_path[0])[0]
            dst_type = g.to_canonical_etype(meta_path[-1])[2]
            if src_type == dst_type:
                src_feature_list = features_dic[src_type]
            else:
                src_feature_list = torch.cat((features_dic[src_type],features_dic[dst_type]),dim = 0)
            dst_feature_list = features_dic[dst_type]

            new_g = self._cached_coalesced_graph[meta_path]

            dst_feature_new_list = self.gat_layers[i](new_g, (src_feature_list, dst_feature_list)).flatten(1)
            semantic_embeddings_dic[dst_type].append(dst_feature_new_list)
            meta_path_list_dic[dst_type].append(meta_path)

        for ntype in g.ntypes:
            # if features_dic[ntype].size()[1] == 300:
            #     self_loop = torch.cat([features_dic[ntype]] * 8, dim=1)
            # else:
            #     self_loop = features_dic[ntype]
            # semantic_embeddings_dic[ntype].append(features_dic[ntype])
            if len(semantic_embeddings_dic[ntype]) == 0:
                # semantic_embeddings_dic[ntype] = [torch.cat([features_dic[ntype]] * int(self.out_size * self.layer_num_heads / self.in_size) , dim = 1)]
                # meta_path_list_dic[ntype] = [(ntype + "_self_loop",)]
                semantic_embeddings_dic[ntype] = None
                continue
            new_embedding = torch.stack(semantic_embeddings_dic[ntype],dim = 1) # [n_node, n_meta_paths, n_features]
            new_embedding = self.semantic_attention_dic[ntype](new_embedding, meta_path_list_dic[ntype])
            # pre_embedding = torch.cat([features_dic[ntype]] * int(self.out_size * self.layer_num_heads / self.in_size) , dim = 1)
            # if ntype == "claim":
            #     update_gate = self.update_gate_claim
            # elif ntype == "evidence":
            #     update_gate = self.update_gate_evidence
            # elif ntype == "verb":
            #     update_gate = self.update_gate_verb
            # elif ntype == "argument":
            #     update_gate = self.update_gate_argument
            # update_gate = update_gate.expand((pre_embedding.shape[0],) + update_gate.shape)
            # retain_gate = 1 - update_gate
            # wandb.log({
            #     "retain_gate_" + str(self.n_layer) + "_" + str(ntype): retain_gate.detach().cpu()
            # })
            # pre_embedding = torch.mul(retain_gate,pre_embedding)
            # new_embedding = torch.mul(update_gate,new_embedding)
            # semantic_embeddings_dic[ntype] = pre_embedding + new_embedding
            #
            semantic_embeddings_dic[ntype] = new_embedding
        return semantic_embeddings_dic

class HAN(nn.Module):
    def __init__(self, meta_paths, ntypes, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, ntypes, in_size, hidden_size, num_heads[0], dropout, n_layer = 0))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, ntypes, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout, n_layer = l))
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size)

    def forward(self, g, features_dic, temp = None):
        for i, gnn in enumerate(self.layers):
            features_dic = gnn(g, features_dic)
        for ntype,feature in features_dic.items():
            if feature == None:continue
            feature = self.predict(feature)
            features_dic[ntype] = feature
        return features_dic
