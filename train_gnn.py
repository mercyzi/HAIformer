from torch.utils.data import DataLoader

from utils import load_data, set_path
from layers import Agent, SymptomEncoderXFMR, MyGAT, PreGnn
import json
from utils import save_pickle, make_dirs, load_pickle
from data_utils import *
from conf import *
import torch.nn.functional as F
import torch.utils.data as Data



# load dataset
train_s, test_s = load_data(train_path), load_data(test_path)
with open(prior_feat_path, encoding='utf-8') as f:
    prior_feat = json.load(f)
record = {}

train_samples = train_s
test_samples = test_s

# muzhi:36; mdd:116; mz10:138
real_prior_feat = prior_feat[:prior_feat_nums]

for i, sample in enumerate(train_samples):
    imp_set = {}
    for k, v in sample['imp_sxs'].items():
        if v == '1':
            sy_sta = k + '-True'
        else:
            sy_sta = k + '-False'
        if sy_sta in real_prior_feat:
            imp_set[k] = v
    sample['imp_key_sxs'] = imp_set


train_size, test_size = len(train_samples), len(test_samples)

# construct symptom & disease vocabulary
sv = SymptomVocab(samples=train_samples, add_special_sxs=True, min_sx_freq=min_sx_freq, max_voc_size=max_voc_size, prior_feat=real_prior_feat)
dv = DiseaseVocab(samples=train_samples)
num_sxs, num_dis = sv.num_sxs, dv.num_dis

# init dataloader
train_ds = SymptomDataset(train_samples, sv, dv, keep_unk, add_tgt_start=True, add_tgt_end=True, train_mode = True)


from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx

graph = create_train_graph(train_ds).to(device)
gnn = PreGnn(hidden_channels=dec_emb_dim).to(device)
optimizer_g = torch.optim.Adam(gnn.parameters(), lr=3e-6, weight_decay=0)
criterion_g = torch.nn.CrossEntropyLoss().to(device)
real_mask = []
for jj in graph.x:
    if len(graph.edge_index[1][graph.edge_index[0] == jj]) != 0:
        real_mask.append(jj)
mask_train = torch.stack(real_mask)
print(mask_train.size()[0])
best_acc=0
for i in range(1000000):
    gnn.train()
    pred = gnn(graph.x, graph.edge_index,graph.x[mask_train])
    loss = criterion_g(pred, graph.x[mask_train]) 
    optimizer_g.zero_grad()
    loss.backward()
    optimizer_g.step()
    if i%1000 ==0:
        print(f"Epoch: {i:03d}, Loss: {loss.item() :.4f}")
        gnn.eval()
        pred = gnn(graph.x , graph.edge_index, graph.x[mask_train])
        max_index = torch.argmax(pred, dim=1)
        correct = max_index.eq(graph.x[mask_train]).sum().item()
        Acc = correct/graph.x[mask_train].size()[0]
        if best_acc < Acc:
            gnn.savegnn(pre_gnn_path)
            best_acc = Acc
            print('saving {} gnn model to {}'.format(best_acc, pre_gnn_path))
        if best_acc > Acc:
            print('Train_gnn over, saving {} gnn model to {}'.format(best_acc, pre_gnn_path))
            assert 0
        print(f"Acc: {Acc :.4f} Best Acc: {best_acc :.4f}")
