# This is all the neural networks.
import os
import torch
import torch.nn as nn
from torch.nn import functional
from torch.distributions import Categorical
from tqdm import tqdm
from activation import MultiheadedAttention
from torch_geometric.nn import GCNConv, GATConv, TransformerConv,SuperGATConv,GATv2Conv
import torch_geometric
import math
from sklearn.metrics.pairwise import cosine_similarity

from data_utils import SymptomVocab, PatientSimulator, device, HumanAgent, create_graph, to_numpy_
from conf import *

weight_cus = 0
# sinusoid position embedding
class PositionalEncoding(nn.Module):

    def __init__(self, emb_dim: int):
        super().__init__()
        self.dropout = nn.Dropout(p=pos_dropout)

        position = torch.arange(pos_max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(pos_max_len, 1, emb_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class Ratio(nn.Module):

    def __init__(self, int_v):
        super().__init__()
        self.exe_ratio = nn.Parameter(torch.empty(1))
        torch.nn.init.constant_(self.exe_ratio, int_v)

    def forward(self, exe_ids):
        return self.exe_ratio * exe_ids


    
# transformer-based decoder
class SymptomDecoderXFMR(nn.Module):

    def __init__(self, sx_embedding, attr_embedding, exe_embedding, num_sxs: int, emb_dim: int, graph):
        super().__init__()

        self.num_sxs = num_sxs

        self.sx_embedding = sx_embedding
        self.attr_embedding = attr_embedding
        self.exe_embedding = exe_embedding
        self.pos_embedding = EmphasizedPositionalEncoding(emb_dim)

        self.decoder = CustomTransformerEncoder(
                d_model=emb_dim,
                nhead=dec_num_heads,
                dim_feedforward=dec_dim_feedforward,
                dropout=dec_dropout)
        
        self.sx_fc = nn.Linear(emb_dim, num_sxs)
        self.graph = graph
        self.ratio = Ratio(1)
        g_data = graph
        if graph is not None:
            self.decoder.EncoderLayer1.gnnmodel.init_graph_dict(g_data.x, g_data.edge_index)
            self.decoder.EncoderLayer2.gnnmodel.init_graph_dict(g_data.x, g_data.edge_index)
            self.decoder.EncoderLayer3.gnnmodel.init_graph_dict(g_data.x, g_data.edge_index)
            self.decoder.EncoderLayer4.gnnmodel.init_graph_dict(g_data.x, g_data.edge_index)

    def forward(self, sx_ids, attr_ids, exe_ids, mask=None, src_key_padding_mask=None):
        if not sx_one_hot and not attr_one_hot:
            inputs = self.sx_embedding(sx_ids) + self.attr_embedding(attr_ids) #+ self.exe_embedding(exe_ids)
        else:
            inputs = torch.cat([self.sx_embedding(sx_ids), self.attr_embedding(attr_ids)], dim=-1)
        if dec_add_pos:
            inputs = self.pos_embedding(inputs, exe_ids)
        self.decoder.EncoderLayer1.exe_ids = exe_ids * ratio1
        self.decoder.EncoderLayer2.exe_ids = exe_ids * ratio2
        self.decoder.EncoderLayer3.exe_ids = exe_ids * ratio3
        self.decoder.EncoderLayer4.exe_ids = exe_ids * ratio4

        self.decoder.cur_sx_ids = sx_ids
        self.decoder.cur_exe_ids = exe_ids
        
        outputs = self.decoder(inputs, mask, src_key_padding_mask)
        
        return outputs

    def get_features(self, outputs):
        features = self.sx_fc(outputs)
        return features

    def compute_entropy(self, features):
        return torch.distributions.Categorical(functional.softmax(features, dim=-1)).entropy().item() / math.log(self.num_sxs)

    def init_repeat_score(self, bsz: int, sv: SymptomVocab, batch: dict = None):
        prob = torch.zeros(bsz, self.num_sxs, device=device)
        prob[:, :sv.num_special] = float('-inf')
        prob[:, sv.end_idx] = float('-inf')
        if exclude_exp:
            assert batch is not None
            for idx in range(bsz):
                for sx in batch['exp_sx_ids'][:, idx]:
                    if sx != sv.pad_idx:
                        prob[idx, sx] = float('-inf')
        if only_key:
            assert batch is not None
            for idx in range(bsz):
                for sx in range(sv.num_special, self.num_sxs):
                    if sx not in sv.prior_sx_attr.keys() and sx not in sv.prior_sx_attr_2.keys():
                        prob[idx, sx] = float('-inf')
        return prob

    @staticmethod
    def update_repeat_score(action, score):
        for act, sc in zip(action, score):
            sc[act.item()] = float('-inf')

    def simulate(self, batch: dict, ps: PatientSimulator, sv: SymptomVocab, hg: HumanAgent, max_turn: int, inference: bool = False):
        # 初始化输入
        _, bsz = batch['exp_sx_ids'].shape
        sx_ids = torch.cat([batch['exp_sx_ids'], ps.init_sx_ids(bsz)])
        
        attr_ids = torch.cat([batch['exp_attr_ids'], ps.init_attr_ids(bsz)])
        exe_ids = torch.cat([batch['exp_exe_ids'], ps.init_attr_ids(bsz)])
        he_value = []
        act_role = []
        # 初始化重复分数，手动将选择特殊symptom的action的概率设置为无穷小
        repeat_score = self.init_repeat_score(bsz, sv, batch)
        
        actions, log_probs = [], []
        if inference:
            hg.change_ability(human_ability)
        else:
            hg.change_ability(1)
        hg.init_batch(batch)
        # 采样 trajectory
        if max_turn > 0:
            for step in range(max_turn):
                
                mask_log_prob = torch.tensor([1 for i in range(bsz)], dtype=torch.bool, device = device)
                mask_valid = torch.tensor([1 for i in range(bsz)], dtype=torch.bool, device = device)
                # 前向传播计算选择每个action的概率
                src_key_padding_mask = sx_ids.eq(sv.pad_idx).transpose(1, 0).contiguous()

                # sx_ids: [seq_len, batch_size = 128]

                # start_id
                output_key_padding_mask = sx_ids.eq(sv.start_idx)

                outputs = self.forward(sx_ids, attr_ids, exe_ids, src_key_padding_mask=src_key_padding_mask)
                # print(outputs.size())
                # print(output_key_padding_mask[0])
                # print(outputs[output_key_padding_mask].size())
                # assert 0
                # outputs: [seq_len, batch_size = 128, emb_dim = 128]
                
                features = self.get_features(outputs[output_key_padding_mask])
                # features: [batch_size = 128, num_sxs]
                
                if inference:
                    # greedy decoding
                    action = (features + repeat_score).argmax(dim=-1)
                    action_confs = to_numpy_(functional.softmax(features + repeat_score, dim = 1).max(-1)[0])
                    if step >= human_step_s and step <= human_step and human_invlove:
                        mask_log_prob, mask_valid = hg.involve(action, sx_ids.permute([1, 0]), attr_ids.permute([1, 0]),action_confs=action_confs)
                        he_value.append(mask_log_prob.count(0))
                        mask_log_prob = torch.tensor(mask_log_prob, dtype=torch.bool, device = device)
                        mask_valid = torch.tensor(mask_valid, dtype=torch.bool, device = device)
                else:
                    # 根据policy网络当前的参数抽样
                    policy = Categorical(functional.softmax(features + repeat_score, dim=-1))
                    action = policy.sample()
                    # 医生在3/4/5/6轮可参与
                    if step >= human_step_s and step <= human_step and human_invlove:
                        mask_log_prob, mask_valid = hg.involve(action, sx_ids.permute([1, 0]), attr_ids.permute([1, 0]))
                        he_value.append(mask_log_prob.count(0))
                        mask_log_prob = torch.tensor(mask_log_prob, dtype=torch.bool, device = device)
                        mask_valid = torch.tensor(mask_valid, dtype=torch.bool, device = device)
                        log_prob = policy.log_prob(action).masked_fill(~mask_log_prob, 0.0) 
                    else:
                        log_prob = policy.log_prob(action)
                    log_probs.append(log_prob)
                # 让已经选择的action再次被解码出的概率为无穷小
                self.update_repeat_score(action, repeat_score)
                # 与病人模拟器进行交互，病人模拟器告知agent病人是否具有该症状
                _, q_attr_ids = ps.answer(action, batch)
                # 更新 transformer 的输入
                sx_ids = torch.cat([sx_ids, action.unsqueeze(dim=0)])
                attr_ids = torch.cat([attr_ids, q_attr_ids.unsqueeze(dim=0)])
                q_exe_ids = torch.tensor([0 if i == True else 1 for i in list(mask_log_prob.cpu())], device=device)
                
                exe_ids = torch.cat([exe_ids, q_exe_ids.unsqueeze(dim=0)])                
                # 记录选择的动作和对数概率（便于之后计算回报和优化）
                actions.append(action)
                act_role.append(mask_log_prob)
                
        else:
            actions.append(torch.tensor([sv.end_idx] * bsz, device=device))
            log_probs.append(torch.tensor([0] * bsz, device=device))
        # 返回整个batch的 trajectory 和对数概率
        # critic_values = torch.stack(critic_values, dim=1)
        si_actions = torch.stack(actions, dim=1)
        act_role = torch.stack(act_role, dim=1)
        si_log_probs = None if inference else torch.stack(log_probs, dim=1)
        si_sx_ids = sx_ids.permute((1, 0))
        si_attr_ids = attr_ids.permute((1, 0))
        return si_actions, si_log_probs, si_sx_ids, si_attr_ids, he_value, act_role

    def inference(self, batch: dict, ps: PatientSimulator, sv: SymptomVocab,  hg: HumanAgent, max_turn: int):
        return self.simulate(batch, ps, sv, hg, max_turn, inference=True)

    def generate(self, ds_loader, ps: PatientSimulator, sv: SymptomVocab, max_turn: int):
        from data_utils import to_list
        ds_sx_ids, ds_attr_ids, ds_labels = [], [], []
        self.eval()
        with torch.no_grad():
            for batch in tqdm(ds_loader):
                _, _, sx_ids, attr_ids = self.inference(batch, ps, sv, max_turn)
                ds_sx_ids.extend(to_list(sx_ids))
                ds_attr_ids.extend(to_list(attr_ids))
                ds_labels.extend(to_list(batch['labels']))
        return ds_sx_ids, ds_attr_ids, ds_labels


class SymptomEncoderXFMR(nn.Module):

    def __init__(self, sx_embedding, attr_embedding, num_sxs, num_dis):
        super().__init__()

        self.num_dis = num_dis
        self.sx_embedding = nn.Embedding(num_sxs, enc_emb_dim, padding_idx=0)
        self.attr_embedding = nn.Embedding(num_attrs, enc_emb_dim, padding_idx=0)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=enc_emb_dim,
                nhead=enc_num_heads,
                dim_feedforward=enc_num_layers,
                dropout=enc_dropout,
                activation='gelu'),
            num_layers=enc_num_layers)

        self.dis_fc = nn.Linear(enc_emb_dim, num_dis, bias=True)

    def forward(self, sx_ids, attr_ids, mask=None, src_key_padding_mask=None):
        if not sx_one_hot and not attr_one_hot:
            inputs = self.sx_embedding(sx_ids) + self.attr_embedding(attr_ids) 
        else:
            inputs = torch.cat([self.sx_embedding(sx_ids), self.attr_embedding(attr_ids)], dim=-1)
        outputs = self.encoder(inputs, mask, src_key_padding_mask)
        return outputs

    # mean pooling feature
    def get_mp_features(self, sx_ids, attr_ids, pad_idx):
        src_key_padding_mask = sx_ids.eq(pad_idx).transpose(1, 0).contiguous()
        # print(sx_ids.shape)
        outputs = self.forward(sx_ids, attr_ids, src_key_padding_mask=src_key_padding_mask)
        # print(outputs.shape)
        seq_len, batch_size, emb_dim = outputs.shape
        mp_mask = (1 - sx_ids.eq(pad_idx).int())
        
        mp_mask_ = mp_mask.unsqueeze(-1).expand(seq_len, batch_size, emb_dim)
        
        avg_outputs = torch.sum(outputs * mp_mask_, dim=0) / torch.sum(mp_mask, dim=0).unsqueeze(-1)
        
        features = self.dis_fc(avg_outputs)
        
        return features

    def predict(self, sx_ids, attr_ids, pad_idx):
        outputs = self.get_mp_features(sx_ids, attr_ids, pad_idx)
        labels = outputs.argmax(dim=-1)
        return labels

    def inference(self, sx_ids, attr_ids, pad_idx):
        return self.simulate(sx_ids, attr_ids, pad_idx, inference=True)

    def compute_entropy(self, features):
        return torch.distributions.Categorical(functional.softmax(features, dim=-1)).entropy().item() / self.num_dis

    @staticmethod
    def compute_max_prob(features):
        return torch.max(functional.softmax(features, dim=-1))


class Agent(nn.Module):

    def __init__(self, num_sxs: int, num_dis: int, sv, graph=None):

        super().__init__()

        if sx_one_hot:
            sx_embedding = nn.Embedding(num_sxs, num_sxs)
            sx_embedding.weight.data = torch.eye(num_sxs)
            sx_embedding.weight.requires_grad = False
            self.sx_embedding = sx_embedding
        else:
            self.sx_embedding = nn.Embedding(num_sxs, dec_emb_dim, padding_idx=0)

        if attr_one_hot:
            attr_embedding = nn.Embedding(num_attrs, num_attrs)
            attr_embedding.weight.data = torch.eye(num_attrs)
            attr_embedding.weight.requires_grad = False
            self.attr_embedding = attr_embedding
        else:
            self.attr_embedding = nn.Embedding(num_attrs, dec_emb_dim, padding_idx=0)

        self.exe_embedding = nn.Embedding(num_executor, dec_emb_dim, padding_idx=0)

        if self.sx_embedding.weight.data.shape[-1] != self.attr_embedding.weight.data.shape[-1]:
            emb_dim = self.sx_embedding.weight.data.shape[-1] + self.attr_embedding.weight.data.shape[-1]
        else:
            emb_dim = dec_emb_dim

        # self.pos_embedding = PositionalEncoding(emb_dim)

        self.symptom_decoder = SymptomDecoderXFMR(
            self.sx_embedding, self.attr_embedding, self.exe_embedding, num_sxs, emb_dim, graph)

        self.symptom_encoder = SymptomEncoderXFMR(
           self.sx_embedding, self.attr_embedding, num_sxs, num_dis
        )
        self.num_sxs = num_sxs
        self.emb_dim = emb_dim
        self.sv = sv

    def forward(self):
        pass

    def init_decoder(self):
        self.symptom_decoder = SymptomDecoderXFMR(
            self.sx_embedding, self.attr_embedding, self.pos_embedding, self.num_sxs, self.emb_dim).to(device)
        
    def load(self, path):
        if os.path.exists(path):
            state_dict = torch.load(path)
            shared_state_dict = {k: v for k, v in state_dict.items() if k in self.state_dict() and 'gnnmodel' not in k}
            self.load_state_dict(shared_state_dict, strict=False)
            if verbose:
                print('loading pre-trained parameters from {} ...'.format(path))

    def save(self, path):
        torch.save(self.state_dict(), path)
        if verbose:
            print('saving best model to {}'.format(path))

    def execute(self, batch: dict, ps: PatientSimulator, sv: SymptomVocab, hg: HumanAgent, max_turn: int, eps: float):
        from data_utils import make_features_xfmr
        _, bsz = batch['exp_sx_ids'].shape
        sx_ids = torch.cat([batch['exp_sx_ids'], ps.init_sx_ids(bsz)])
        attr_ids = torch.cat([batch['exp_attr_ids'], ps.init_attr_ids(bsz)])
        repeat_score = self.symptom_decoder.init_repeat_score(bsz, sv, batch)
        hg.init_batch(batch)
        # print(batch)
        he_value, he_values = [], []

        for step in range(max_turn + 1):
            # 每一个step，先观察 encoder 的 entropy 是否已经有足够的 confidence 给出诊断结果
            si_sx_ids = sx_ids.clone().permute((1, 0))
            # 最新动作是否为结束符号
            act_id = si_sx_ids[0][-1].item()

            si_attr_ids = attr_ids.clone().permute((1, 0))
            si_sx_feats, si_attr_feats = make_features_xfmr(
                sv, batch, si_sx_ids, si_attr_ids, merge_act=False, merge_si=True)
            
            dc_outputs = self.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
            print(dc_outputs.shape)
            assert 0
            prob = self.symptom_encoder.compute_max_prob(dc_outputs).item()
            if prob > eps or step == max_turn or act_id == sv.end_idx:
                is_success = batch['labels'].eq(dc_outputs.argmax(dim=-1)).item()
                max_prob = prob
                
                return step, is_success, max_prob, he_value
            # 再观察 decoder 的 entropy 是否已经有足够的 confidence 给出诊断结果
            src_key_padding_mask = sx_ids.eq(sv.pad_idx).transpose(1, 0).contiguous()
            outputs = self.symptom_decoder.forward(sx_ids, attr_ids, src_key_padding_mask=src_key_padding_mask)
            features = self.symptom_decoder.get_features(outputs[-1])
            action = (features + repeat_score).argmax(dim=-1)
            if step >= 2 and step <= 5 and human_invlove:
                mask_log_prob = hg.involve(action, sx_ids.permute([1, 0]), attr_ids.permute([1, 0]))
                he_value.append(mask_log_prob.count(0))
            self.symptom_decoder.update_repeat_score(action, repeat_score)
            _, q_attr_ids = ps.answer(action, batch)
            sx_ids = torch.cat([sx_ids, action.unsqueeze(dim=0)])
            attr_ids = torch.cat([attr_ids, q_attr_ids.unsqueeze(dim=0)])        
        assert 0
    


# sinusoid position embedding
class EmphasizedPositionalEncoding(nn.Module):

    def __init__(self, emb_dim: int):
        super().__init__()
        self.dropout = nn.Dropout(p=pos_dropout)

        position = torch.arange(pos_max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(pos_max_len, 1, emb_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, exe_ids):
        # 获取原始位置编码
        original_pe = self.pe[:x.size(0)]
        original_pe = original_pe.expand(x.shape)
        # 创建一个强调子序列位置的 mask
        mask = torch.zeros_like(original_pe)
        # 获取非零值的索引
        nonzero_indices = torch.nonzero(exe_ids)    
        for index in nonzero_indices:
            i, j = index
            mask[i, j] = 1  # 假设需要强调索引的子序列位置
        # 修改位置编码值以强调子序列的重要性
        emphasized_pe = original_pe + emphasis_factor * original_pe * mask
        x = x + emphasized_pe
        return self.dropout(x)

# gnn model
class MyGAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(MyGAT, self).__init__()
        self.user_emb = torch.nn.Embedding(gnn_nodes_num, hidden_channels)
        
        self.GAT1 = TransformerConv(hidden_channels, hidden_channels)
        
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
 

    def forward(self, x, edge_index):
        x = self.GAT1(x=self.user_emb(x), edge_index=edge_index)
        
        
        return x
# pretrain gnn model
class PreGnn(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(PreGnn, self).__init__()
        self.gnn = MyGAT(hidden_channels=hidden_channels)
       
        # self.linear0 = nn.Linear(hidden_channels, hidden_channels)
        self.linear1 = nn.Linear(hidden_channels, gnn_nodes_num)
 

    def forward(self, x, edge_index, y):
        x = self.gnn(x, edge_index)
        
        new_x = []
        for x_id in y:
            x_neighbor = edge_index[1][edge_index[0] == x_id]
            new_x.append(x[x_neighbor].sum(0))           
        new_x = torch.stack(new_x).to(device)
        
        x = self.linear1(new_x)
        
        return functional.softmax(x,dim=-1)
    
    def savegnn(self, path):
        torch.save(self.gnn.state_dict(), path)

# compute sim of sxs
class ComSimModel(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(ComSimModel, self).__init__()
        self.gnn = MyGAT(hidden_channels=hidden_channels)
        self.linear0 = nn.Linear(hidden_channels, hidden_channels)
        self.linear1 = nn.Linear(hidden_channels, hidden_channels)
        self.linear = nn.Linear(2 * hidden_channels, 1, bias=False)
        self.relu = nn.Sigmoid()
        self.graph_dict = []
 
    def init_graph_dict(self, x, edge_index):
        x = self.gnn(x, edge_index)
        self.graph_dict = x.detach()
        

    def forward(self,batch_nodes,batch_nodes_exe, src):

        tgt_len, bsz, embed_dim = src.shape
        new_x = []
        gate_x = []
        batch_nodes = batch_nodes.cpu()
        batch_nodes_exe = batch_nodes_exe.cpu()
        
        for i in range(batch_nodes.size()[0]):
            if 1 in batch_nodes_exe[i]:
                arrays = list(batch_nodes_exe[i])
                doc_id = tgt_len - arrays[::-1].index(1) - 1
                new_x.append(self.graph_dict[batch_nodes[i][doc_id]].view(1,embed_dim))
                gate_x.append(torch.ones([1]))
            else:
                new_x.append(torch.zeros([1,embed_dim]))
                gate_x.append(torch.zeros([1]))
        new_x = torch.stack(new_x).to(device)
        gate_x = torch.stack(gate_x).to(device)
        
        # x = new_x
        x = self.linear0(new_x)
        
        # 计算余弦相似度V
        cosine_sim_matrix = torch.matmul(x, src.permute(1, 2, 0))/torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        
        # gate_value = self.relu(self.linear(torch.cat((x, src.permute(1, 0, 2)), dim=-1)))
        
        return cosine_sim_matrix, gate_x

        # return functional.softmax(torch.matmul(x, x.transpose(-2, -1))/torch.tensor(node_dim, dtype=torch.float32),dim=-2)
    
    def loadgnn(self, path):
        self.gnn.load_state_dict(torch.load(path))
        
# 自定义的TransformerEncoderLayer
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        # self.self_attn = MultiHeadedAttention(d_model, nhead)
        self.self_attn = MultiheadedAttention(d_model, nhead)
        
        self.gnnmodel = ComSimModel(hidden_channels=dec_emb_dim)
        # self.gnnmodel.loadgnn(pre_gnn_path)
        for param in self.gnnmodel.gnn.parameters():
            param.requires_grad = False
        # self.self_gate_attn = nn.MultiheadAttention(d_model, nhead)
        self.gnnmodel = self.gnnmodel
        self.nhead = nhead
        self.norm_first = False
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = functional.gelu


        # Gate 控制自定义权重的应用
        self.gate = nn.Linear(d_model, 1)
        # 自定义权重的应用
        self.exe_ids = None
        self.custom_weights = nn.Parameter(torch.randn(d_model))
        # graph构造 
        self.graph = None

    def forward(self, src, src_mask=None, src_key_padding_mask = None, is_causal: bool = False, graph_score = None, gate_value=None):

        src_key_padding_mask = functional._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=functional._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )
        
        x = src
        
        # 自定义的权重，这里使用随机生成的权重，你可以根据任务和需求调整
        if self.exe_ids is not None:
            custom_weights = self.exe_ids
        else:
            custom_weights = None
        # 多头自注意力
        attn_output = self.dropout3(self.self_attn(x, x, x, attn_mask = src_mask, key_padding_mask=src_key_padding_mask, need_weights=False, exe_ids = custom_weights, graph_score = graph_score,gate_value=gate_value)[0])

        modified_attn_output = attn_output #+ gate_value * custom_weights

        x = self.norm1(modified_attn_output)
        x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x,attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
class CustomTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoder, self).__init__()
        self.EncoderLayer1 = CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout)
        self.EncoderLayer2 = CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout)
        self.EncoderLayer3 = CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout)
        self.EncoderLayer4 = CustomTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout)
        self.cur_sx_ids = None
        self.cur_exe_ids = None
        self.pooling = nn.AdaptiveAvgPool1d(1)
    def forward(self, src, src_mask=None, src_key_padding_mask = None, is_causal: bool = False):

        seq_len = src.size()[0]
        bsz = src.size()[1]
        node_dim = src.size()[2]
        batch_nodes = self.cur_sx_ids.permute([1, 0])
        batch_nodes_exe = self.cur_exe_ids.permute([1, 0])

        # graph_score = None
        graph_score, gate_value = self.EncoderLayer1.gnnmodel(batch_nodes,batch_nodes_exe, src)
        layer1_val = self.EncoderLayer1(src, src_mask, src_key_padding_mask, graph_score= graph_score, gate_value=gate_value)

        graph_score, gate_value = self.EncoderLayer2.gnnmodel(batch_nodes,batch_nodes_exe,layer1_val)
        layer2_val = self.EncoderLayer2(layer1_val, src_mask, src_key_padding_mask, graph_score= graph_score, gate_value=gate_value)

        graph_score, gate_value = self.EncoderLayer3.gnnmodel(batch_nodes,batch_nodes_exe,layer2_val)
        layer3_val = self.EncoderLayer3(layer2_val, src_mask, src_key_padding_mask, graph_score= graph_score, gate_value=gate_value)

        graph_score, gate_value = self.EncoderLayer4.gnnmodel(batch_nodes,batch_nodes_exe,layer3_val)
        layer4_val = self.EncoderLayer4(layer3_val, src_mask, src_key_padding_mask, graph_score= graph_score, gate_value=gate_value)

        return layer4_val
    


