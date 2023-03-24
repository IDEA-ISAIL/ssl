import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np
import copy
from .base import Model
from torch_geometric.typing import Tensor, Adj



def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.layer = GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, item_embs, entity_embs, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer(x, y, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x
    
    def forward_relation(self, item_embs, entity_embs, w_r, adj):
        x = F.dropout(item_embs, self.dropout, training=self.training)
        y = F.dropout(entity_embs, self.dropout, training=self.training)
        x = self.layer.forward_relation(x, y, w_r, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x




class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(2*out_features, out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward_relation(self, item_embs, entity_embs, relations, adj):
        # item_embs: N, dim
        # entity_embs: N, e_num, dim
        # relations: N, e_num, r_dim
        # adj: N, e_num
        
        # N, e_num, dim
        Wh = item_embs.unsqueeze(1).expand(entity_embs.size())
        # N, e_num, dim
        We = entity_embs
        a_input = torch.cat((Wh,We),dim=-1) # (N, e_num, 2*dim)
        # N,e,2dim -> N,e,dim
        e_input = torch.multiply(self.fc(a_input), relations).sum(-1) # N,e
        e = self.leakyrelu(e_input) # (N, e_num)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def forward(self, item_embs, entity_embs, adj):
        Wh = torch.mm(item_embs, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        We = torch.matmul(entity_embs, self.W) # entity_embs: (N, e_num, in_features), We.shape: (N, e_num, out_features)
        a_input = self._prepare_cat(Wh, We) # (N, e_num, 2*out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # (N, e_num)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # N, e_num
        # (N, 1, e_num) * (N, e_num, out_features) -> N, out_features
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), entity_embs).squeeze()
        h_prime = entity_emb_weighted+item_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_cat(self, Wh, We):
        Wh = Wh.unsqueeze(1).expand(We.size()) # (N, e_num, out_features)
        return torch.cat((Wh, We), dim=-1) # (N, e_num, 2*out_features)


    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class ModelKGCL(nn.Module):
    r"""The full model to train the encoder.

    Args:
        
    """
    def __init__(self, 
                config: dict,
                dataset,
                 kg_dataset):
        super(ModelKGCL, self).__init__()
        self.world = config
        self.config = config.config
        self.dataset = dataset
        self.kg_dataset = kg_dataset
        self.__init_weight()
        self.gat = GAT(self.latent_dim, self.latent_dim, dropout=0.4, alpha=0.2).train()
    
    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.num_entities = self.kg_dataset.entity_count
        self.num_relations = self.kg_dataset.relation_count

        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']

        # Lihui: what's the difference between user, item, entity
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # item and kg entity
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_entity = torch.nn.Embedding(
            num_embeddings=self.num_entities+1, embedding_dim=self.latent_dim)
        self.embedding_relation = torch.nn.Embedding(
            num_embeddings=self.num_relations+1, embedding_dim=self.latent_dim)
        
        # relation weights, a 3d matrix
        self.W_R = nn.Parameter(torch.Tensor(self.num_relations, self.latent_dim, self.latent_dim))
        # lihui: what's nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        if self.config['pretrain'] == 0:
            print('use NORMAL distribution UI')
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print('use NORMAL distribution ENTITY')
            nn.init.normal_(self.embedding_entity.weight, std=0.1)
            nn.init.normal_(self.embedding_relation.weight, std=0.1)
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        # Lihui: what's the purpose of the sparse graph
        self.Graph = self.dataset.getSparseGraph()
        self.kg_dict, self.item2relations = self.kg_dataset.get_kg_dict(self.num_items)
        print(f"KGCL is ready to go!")


    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def view_computer_all(self, g_droped, kg_droped):
        """
        propagate methods for contrastive lightGCN
        Lihui: don't understand
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(kg_droped)
        all_emb = torch.cat([users_emb, items_emb])  # 91457 x 64
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


    def view_computer_ui(self, g_droped):
        """
        propagate methods for contrastive lightGCN
        Lihui: don't understand
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


    def computer(self):
        """
        propagate methods for lightGCN
        Lihui: don't understand
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_kg(self.kg_dict)
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        # this is just getting embedding for regularization
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        # mean or sum
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        if(torch.isnan(loss).any().tolist()):
            print("user emb")
            print(userEmb0)
            print("pos_emb")
            print(posEmb0)
            print("neg_emb")
            print(negEmb0)
            print("neg_scores")
            print(neg_scores)
            print("pos_scores")
            print(pos_scores)
            return None
        # loss corresponds to the BPR loss for recommendation task.
        # reg_loss computes the L2 regularization for batch user and item embeddings. 
        return loss, reg_loss
       
       
    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.embedding_relation(r)                 # (kg_batch_size, relation_dim)
        h_embed = self.embedding_item(h)              # (kg_batch_size, entity_dim)
        pos_t_embed = self.embedding_entity(pos_t)      # (kg_batch_size, entity_dim)
        neg_t_embed = self.embedding_entity(neg_t)      # (kg_batch_size, entity_dim)
        # Equation (1)
        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)     # (kg_batch_size)
        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        # loss = kg_loss
        return loss


    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.embedding_relation(r)                 # (kg_batch_size, relation_dim)
        W_r = self.W_R[r]                                # (kg_batch_size, entity_dim, relation_dim)

        h_embed = self.embedding_item(h)              # (kg_batch_size, entity_dim)
        pos_t_embed = self.embedding_entity(pos_t)      # (kg_batch_size, entity_dim)
        neg_t_embed = self.embedding_entity(neg_t)      # (kg_batch_size, entity_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)             # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)     # (kg_batch_size, relation_dim)

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)     # (kg_batch_size)

        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        # loss = kg_loss
        return loss

    # use GNN to learn the KG embedding, what's the difference between item and entity
    def cal_item_embedding_gat(self, kg:dict):
        item_embs = self.embedding_item(torch.IntTensor(list(kg.keys())).to(self.world.device)) #item_num, emb_dim
        item_entities = torch.stack(list(kg.values())) # item_num, entity_num_each
        entity_embs = self.embedding_entity(item_entities) # item_num, entity_num_each, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities!=self.num_entities, torch.ones_like(item_entities), torch.zeros_like(item_entities)).float()
        return self.gat(item_embs, entity_embs, padding_mask)
    
    def cal_item_embedding_rgat(self, kg:dict):
        item_embs = self.embedding_item(torch.IntTensor(list(kg.keys())).to(self.world.device)) #item_num, emb_dim
        item_entities = torch.stack(list(kg.values())) # item_num, entity_num_each
        item_relations = torch.stack(list(self.item2relations.values()))
        entity_embs = self.embedding_entity(item_entities) # item_num, entity_num_each, emb_dim
        relation_embs = self.embedding_relation(item_relations) # item_num, entity_num_each, emb_dim
        # w_r = self.W_R[relation_embs] # item_num, entity_num_each, emb_dim, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities!=self.num_entities, torch.ones_like(item_entities), torch.zeros_like(item_entities)).float()
        return self.gat.forward_relation(item_embs, entity_embs, relation_embs, padding_mask)


    def cal_item_embedding_from_kg(self, kg: dict):
        if kg is None:
            kg = self.kg_dict

        if(self.world.kgcn=="GAT"):
            return self.cal_item_embedding_gat(kg)
        elif self.world.kgcn=="RGAT":
            return self.cal_item_embedding_rgat(kg)
        elif(self.world.kgcn=="MEAN"):
            return self.cal_item_embedding_mean(kg)
        elif(self.world.kgcn=="NO"):
            return self.embedding_item.weight


    def cal_item_embedding_mean(self, kg: dict):
        item_embs = self.embedding_item(torch.IntTensor(list(kg.keys())).to(self.world.device)) #item_num, emb_dim
        item_entities = torch.stack(list(kg.values())) # item_num, entity_num_each
        entity_embs = self.embedding_entity(item_entities) # item_num, entity_num_each, emb_dim
        # item_num, entity_num_each
        padding_mask = torch.where(item_entities!=self.num_entities, torch.ones_like(item_entities), torch.zeros_like(item_entities)).float()
        # padding为0
        entity_embs = entity_embs * padding_mask.unsqueeze(-1).expand(entity_embs.size())
        # item_num, emb_dim
        entity_embs_sum = entity_embs.sum(1)
        entity_embs_mean = entity_embs_sum / padding_mask.sum(-1).unsqueeze(-1).expand(entity_embs_sum.size())
        # replace nan with zeros
        entity_embs_mean = torch.nan_to_num(entity_embs_mean)
        # item_num, emb_dim
        return item_embs+entity_embs_mean


    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
