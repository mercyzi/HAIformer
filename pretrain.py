from torch.utils.data import DataLoader

from utils import load_data
import json
from layers import Agent
from utils import make_dirs,write_data
import torch.optim.lr_scheduler as lr_scheduler
from data_utils import *
from conf import *

# load dataset
train_s, test_s = load_data(train_path), load_data(test_path)
with open(prior_feat_path, encoding='utf-8') as f:
    prior_feat = json.load(f)
record = {}
train_samples = train_s
test_samples = test_s

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
for sample in test_samples:
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

sv = SymptomVocab(samples=train_samples, add_special_sxs=True, min_sx_freq=min_sx_freq, max_voc_size=max_voc_size, prior_feat=real_prior_feat)
dv = DiseaseVocab(samples=train_samples)

num_sxs, num_dis = sv.num_sxs, dv.num_dis

if data_aug is True:
    train_samples = data_augmentation(train_samples, real_prior_feat)
train_ds = SymptomDataset(train_samples, sv, dv, keep_unk=False, add_tgt_start=True, add_tgt_end=True)
train_ds_loader = DataLoader(train_ds, batch_size=train_bsz, num_workers=num_workers, shuffle=True, collate_fn=lm_collater)

test_ds = SymptomDataset(test_samples, sv, dv, keep_unk=False, add_tgt_start=True, add_tgt_end=True)
test_ds_loader = DataLoader(test_ds, batch_size=test_bsz, num_workers=num_workers, shuffle=False, collate_fn=lm_collater)

model = Agent(num_sxs, num_dis, sv).to(device)

si_criterion = torch.nn.CrossEntropyLoss(ignore_index=sv.pad_idx).to(device)
dc_criterion = torch.nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=pt_learning_rate)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
make_dirs([best_pt_path, last_pt_path])


print('pre-training...')
int_model = Agent(num_sxs, num_dis, sv).to(device)
# best_model.load(best_pt_path)
all_best_acc = pt_learning_save
for i in range(1):
    best_acc = 0
    model.load_state_dict(int_model.state_dict())
    for epoch in range(pt_train_epochs):
        # break
        train_loss, train_si_loss, train_dc_loss = [], [], []
        train_num_hits, test_num_hits = 0, 0
        model.train()
        for batch in train_ds_loader:
            # break
            sx_ids, attr_ids, exe_ids, labels = batch['sx_ids'], batch['attr_ids'], batch['exe_ids'], batch['labels']
            seq_len, bsz = sx_ids.shape
            shift_sx_ids = torch.cat([sx_ids[1:], torch.zeros((1, bsz), dtype=torch.long, device=device)], dim=0)
            # symptom inquiry
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
            si_outputs = model.symptom_decoder.get_features(model.symptom_decoder(sx_ids, attr_ids, exe_ids, mask=mask))
            si_loss = si_criterion(si_outputs.view(-1, num_sxs), shift_sx_ids.view(-1))
            
            # disease classification
            si_sx_feats, si_attr_feats = make_features_xfmr(
                sv, batch, sx_ids.permute(1, 0), attr_ids.permute(1, 0), merge_act=False, merge_si=True)
            dc_outputs = model.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
            dc_loss = dc_criterion(dc_outputs, batch['labels'])
            loss = si_loss + dc_loss * pt_ratio
            # record
            train_loss.append(loss.item())
            train_si_loss.append(si_loss.item())
            train_dc_loss.append(dc_loss.item())
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_num_hits += torch.sum(batch['labels'].eq(dc_outputs.argmax(dim=-1))).item()
        scheduler.step()    
        train_acc = train_num_hits / (train_size)
        model.eval()
        for batch in test_ds_loader:
            sx_ids, attr_ids, labels = batch['sx_ids'], batch['attr_ids'], batch['labels']
            si_sx_feats, si_attr_feats = make_features_xfmr(
                sv, batch, sx_ids.permute(1, 0), attr_ids.permute(1, 0), merge_act=False, merge_si=True)
            dc_outputs = model.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
            test_num_hits += torch.sum(batch['labels'].eq(dc_outputs.argmax(dim=-1))).item()
        test_acc = test_num_hits / test_size
        if test_acc > best_acc and test_acc > all_best_acc:
            best_acc = test_acc
            all_best_acc = test_acc
            
            model.save(best_pt_path)
            print('best acc: {}'.format(round(all_best_acc, digits)))
        print('epoch: {}, train total/si/dc loss: {}/{}/{}, train/test/best/all_best acc: {}/{}/{}/{}'.format(
            epoch + 1, np.round(np.mean(train_loss), digits), np.round(np.mean(train_si_loss), digits),
            np.round(np.mean(train_dc_loss), digits), round(train_acc, digits),
            round(test_acc, digits), round(best_acc, digits), round(all_best_acc, digits)))
# model.save(last_pt_path)
