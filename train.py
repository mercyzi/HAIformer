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

# construct symptom & disease vocabulary
sv = SymptomVocab(samples=train_samples, add_special_sxs=True, min_sx_freq=min_sx_freq, max_voc_size=max_voc_size, prior_feat=real_prior_feat)
dv = DiseaseVocab(samples=train_samples)
num_sxs, num_dis = sv.num_sxs, dv.num_dis

# init dataloader
train_ds = SymptomDataset(train_samples, sv, dv, keep_unk, add_tgt_start=True, add_tgt_end=True, train_mode = True)
train_ds_loader = DataLoader(train_ds, batch_size=train_bsz, num_workers=num_workers, shuffle=True, collate_fn=pg_collater)

test_ds = SymptomDataset(test_samples, sv, dv, keep_unk, add_tgt_start=True, add_tgt_end=True)
test_ds_loader = DataLoader(test_ds, batch_size=len(test_samples), num_workers=num_workers, shuffle=False, collate_fn=pg_collater)

# compute disease-symptom co-occurrence matrix
dscom = compute_dscom(train_samples, sv, dv)

# init reward distributor
rd = RewardDistributor(sv, dv, dscom)

# init human agent
hg = HumanAgent(sv, human_ability)

# init patient simulator
ps = PatientSimulator(sv)

from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx

graph = create_train_graph(train_ds)#.to(device)

print('training...')

mtra_path = 'saved/{}/tra_{}.pt'.format(train_dataset, exp_name)
results = []

for max_turn in range(num_turns, num_turns-1, -1):
    # for max_turn in range(1, num_turns + 1):
    recall = []
    Dxacc = []
    human_turn = []
    all_turn = []
    for num in range(num_repeats):
        # init epoch recorder
        train_sir = SIRecorder(num_samples=len(train_ds), num_imp_sxs=compute_num_sxs(train_samples, sv), digits=digits)
        test_sir = SIRecorder(num_samples=len(test_ds), num_imp_sxs=compute_num_sxs(test_samples, sv), digits=digits)
        # init agent
        model = Agent(num_sxs, num_dis, sv, graph).to(device)
        # load parameters from pre-trained models if exits
        model.load(best_pt_path)
        dc_model = Agent(num_sxs, num_dis, sv, graph).to(device)
        best_acc_model = Agent(num_sxs, num_dis, sv, graph).to(device)
        dc_model.load(best_pt_path)
        dc_model.eval()
        # init optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
        # compute loss of disease classification
        criterion = torch.nn.CrossEntropyLoss().to(device)
        # model path
        rec_model_path = set_path('rec_model', train_dataset, exp_name, max_turn, num, num_repeats)
        acc_model_path = set_path('acc_model', train_dataset, exp_name, max_turn, num, num_repeats)
        metric_model_path = set_path('metric_model', train_dataset, exp_name, max_turn, num, num_repeats)
        log_path = set_path('sir', train_dataset, exp_name, max_turn, num, num_repeats)
        make_dirs([rec_model_path, acc_model_path, metric_model_path, log_path])
        # start
        epochs = train_epochs if max_turn > 0 else warm_epoch
        for epoch in range(epochs):
            num_hits_train, num_hits_test = 0, 0
            # training
            for batch in train_ds_loader:
                np_batch = to_numpy(batch)
                # symptom inquiry
                model.train()
                for param in model.symptom_decoder.ratio.parameters():
                    param.requires_grad = False
                si_actions, si_log_probs, si_sx_ids, si_attr_ids, he_value, act_role = model.symptom_decoder.simulate(batch, ps, sv, hg, max_turn)
                # make features
                si_sx_feats, si_attr_feats = make_features_xfmr(
                    sv, batch, si_sx_ids, si_attr_ids,merge_act=False, merge_si=True)
                # make diagnosis
                dc_outputs = model.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
                diag_hits = batch['labels'].eq(dc_outputs.argmax(dim=-1))
                # compute reward of each step of symptom recovery
                si_rewards = rd.compute_sr_reward(si_actions, np_batch, epoch, he_value, act_role, train_sir, diag_hits)
                loss = - torch.sum(si_log_probs * to_tensor_(si_rewards))
                # compute the gradient and optimize the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # assert 0
            print("traing dc:"+str(epoch))
            if epoch > 19:
                optimizer_dc = torch.optim.Adam(model.parameters(), lr=learning_rate)
                model.symptom_encoder.load_state_dict(dc_model.symptom_encoder.state_dict())
                total_iterations = train_dc_epoch
                id = 0
                cur_accs = []
                while id < total_iterations:
                    if len(Dxacc) >0 and id%2 == 0:
                        print('recall: {} -> {}'.format(round((sum(recall) / len(recall)),4),recall))
                        print('acc: {} -> {}'.format(round((sum(Dxacc) / len(Dxacc)),4),Dxacc))
                        print('HE: {} -> {}'.format(round((sum(human_turn) / len(human_turn)),4),human_turn))
                        print('turn: {} -> {}'.format(round((sum(all_turn) / len(all_turn)),4),all_turn))
                        print('-' * 100)
                    num_hits_train = 0
                    num_hits_test = 0
                    for batch in train_ds_loader:
                        model.eval()
                        with torch.no_grad():
                            _, _, si_sx_ids, si_attr_ids, _,_ = model.symptom_decoder.inference(batch, ps, sv, hg, max_turn)
                        model.train()
                        si_sx_feats, si_attr_feats = make_features_xfmr(
                            sv, batch, si_sx_ids, si_attr_ids, merge_act=False, merge_si=True)
                        dc_outputs = model.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
                        dc_loss = criterion(dc_outputs, batch['labels'])
                        num_hits_train += torch.sum(batch['labels'].eq(dc_outputs.argmax(dim=-1))).item()
                        optimizer_dc.zero_grad()
                        dc_loss.backward()
                        optimizer_dc.step()
                    train_acc = num_hits_train / (train_size)
                    train_sir.update_acc(epoch * 100 + id, train_acc)
                    if verbose:
                        train_sir.epoch_summary(epoch * 100 + id)
                    # evaluation
                    model.eval()
                    random.seed(12354)
                    with torch.no_grad():
                        for batch in test_ds_loader:
                            np_batch = to_numpy(batch)
                            # symptom inquiry
                            si_actions, _, si_sx_ids, si_attr_ids, he_value, act_role = model.symptom_decoder.inference(batch, ps, sv, hg, max_turn)
                            # each seq have seq disease result
                            each_sx_feats, each_attr_feats = make_features_xfmr_seq(
                                sv, batch, si_sx_ids, si_attr_ids, merge_act=False, merge_si=True)
                            early_stop = model.symptom_encoder.get_mp_features(each_sx_feats, each_attr_feats, sv.pad_idx)
                            early_stop = F.softmax(early_stop, dim = 1).max(-1)[0]
                            # compute reward of each step of symptom recovery
                            _ = rd.compute_sr_reward(si_actions, np_batch, epoch * 100 + id, he_value, act_role, test_sir, early_stop=early_stop)
                            # make features
                            si_sx_feats, si_attr_feats = make_features_xfmr(
                                sv, batch, si_sx_ids, si_attr_ids, merge_act=False, merge_si=True)
                            # make diagnosis
                            dc_outputs = model.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
                            num_hits_test += torch.sum(batch['labels'].eq(dc_outputs.argmax(dim=-1))).item()
                    test_acc = num_hits_test / test_size
                    random.seed(None)
                    test_sir.update_acc(epoch * 100 + id, test_acc)
                    if verbose:
                        test_sir.epoch_summary(epoch * 100 + id)
                    cur_rec, best_rec, _, _, cur_acc, best_acc, _, _, cur_met, best_met, _, _,_,_,_= test_sir.report(epoch * 100 + id, digits, alpha, verbose)
                    cur_accs.append(cur_acc)
                    if cur_acc > 0.9 and total_iterations< 45:
                        total_iterations += 5
                    if cur_acc >= best_acc:
                        model.save(acc_model_path)  # save the model with best accuracy
                        best_acc_model.symptom_encoder.load_state_dict(model.symptom_encoder.state_dict())
                        for param in best_acc_model.parameters():
                            param.requires_grad = False
                    if verbose:
                        print('-' * 100)
                    id += 1
                model.symptom_encoder.load_state_dict(best_acc_model.symptom_encoder.state_dict())
        # end training
        _, best_rec, best_rec_epoch, best_rec_acc, _, best_acc, best_acc_epoch, best_acc_rec, _, _, best_met_epoch, best_met_rec, best_met_acc, best_acc_HE, best_acc_turn = test_sir.report(epochs * 101, digits, alpha, verbose)
        result = {
            'max_turn': max_turn,
            'num': num,
            'best_rec_epoch': best_rec_epoch,
            'best_rec': round(best_rec, digits),
            'best_rec_acc': round(best_rec_acc, digits),
            'best_acc_epoch': best_acc_epoch,
            'best_acc_rec': round(best_acc_rec, digits),
            'best_acc': round(best_acc, digits),
            'best_met_epoch': best_met_epoch,
            'best_met_rec': round(best_met_rec, digits),
            'best_met_acc': round(best_met_acc, digits),
        }
        recall.append(best_acc_rec)
        Dxacc.append(best_acc)
        human_turn.append(best_acc_HE)
        all_turn.append(best_acc_turn)
        print(result)
        results.append(result)
        save_pickle((train_sir, test_sir), log_path)
        save_pickle(results, mtra_path)
    print('recall: {} -> {}'.format(round((sum(recall) / len(recall)),4),recall))
    print('acc: {} -> {}'.format(round((sum(Dxacc) / len(Dxacc)),4),Dxacc))
    print('HE: {} -> {}'.format(round((sum(human_turn) / len(human_turn)),4),human_turn))
    print('turn: {} -> {}'.format(round((sum(all_turn) / len(all_turn)),4),all_turn))
