import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from collections import defaultdict
from scipy.stats import truncnorm
from collections import deque
import operator
import math
import random
import torch_geometric.data as data
import copy
# from torch_geometric.loader import DataLoader

from conf import *

device = torch.device('cuda:{}'.format(device_num) if torch.cuda.is_available() else 'cpu')


class SymptomVocab:

    def __init__(self, samples: list = None, add_special_sxs: bool = False,
                 min_sx_freq: int = None, max_voc_size: int = None, prior_feat: list = None):

        # sx is short for symptom
        self.sx2idx = {}  # map from symptom to symptom id
        self.idx2sx = {}  # map from symptom id to symptom
        self.sx2count = {}  # map from symptom to symptom count
        self.num_sxs = 0  # number of symptoms
        self.prior_sx_attr = {} # key symptoms with True
        self.prior_sx_attr_2 = {} # key symptoms with False
        self.no_key_sx = []


        # symptom attrs
        self.SX_ATTR_PAD_IDX = 0  # symptom attribute id for PAD
        self.SX_ATTR_POS_IDX = 1  # symptom attribute id for YES
        self.SX_ATTR_NEG_IDX = 2  # symptom attribute id for NO
        self.SX_ATTR_NS_IDX = 3  # symptom attribute id for NOT SURE
        self.SX_ATTR_NM_IDX = 4  # symptom attribute id for NOT MENTIONED

        # symptom askers
        self.SX_EXE_PAD_IDX = 0  # PAD
        self.SX_EXE_AI_IDX = 1  # AI
        self.SX_EXE_DOC_IDX = 2  # Human

        self.SX_ATTR_MAP = {  # map from symptom attribute to symptom attribute id
            '0': self.SX_ATTR_NEG_IDX,
            '1': self.SX_ATTR_POS_IDX,
            '2': self.SX_ATTR_NS_IDX,
        }

        self.SX_ATTR_MAP_INV = {
            self.SX_ATTR_NEG_IDX: '0',
            self.SX_ATTR_POS_IDX: '1',
            self.SX_ATTR_NS_IDX: '2',
        }

        # special symptoms
        self.num_special = 0  # number of special symptoms
        self.special_sxs = []

        # vocabulary
        self.min_sx_freq = min_sx_freq  # minimal symptom frequency
        self.max_voc_size = max_voc_size  # maximal symptom size

        # add special symptoms
        if add_special_sxs:  # special symptoms
            self.SX_PAD = '[PAD]'
            self.SX_START = '[START]'
            self.SX_END = '[END]'
            self.SX_UNK = '[UNKNOWN]'
            self.SX_CLS = '[CLS]'
            self.SX_MASK = '[MASK]'
            self.special_sxs.extend([self.SX_PAD, self.SX_START, self.SX_END, self.SX_UNK, self.SX_CLS, self.SX_MASK])
            self.sx2idx = {sx: idx for idx, sx in enumerate(self.special_sxs)}
            self.idx2sx = {idx: sx for idx, sx in enumerate(self.special_sxs)}
            self.num_special = len(self.special_sxs)
            self.num_sxs += self.num_special


        # update vocabulary
        if samples is not None:
            if not isinstance(samples, tuple):
                samples = (samples,)
            num_samples = 0
            for split in samples:
                num_samples += self.update_voc(split)
            print('symptom vocabulary constructed using {} split and {} samples '
                  '({} symptoms with {} special symptoms)'.
                  format(len(samples), num_samples, self.num_sxs - self.num_special, self.num_special))

        # trim vocabulary
        self.trim_voc()

        # prior_feat
        if prior_feat is not None:
            for sx in self.sx2idx.keys():
                if sx + '-True' in prior_feat :
                    self.prior_sx_attr[self.sx2idx[sx]] = self.SX_ATTR_POS_IDX
                if sx + '-False' in prior_feat:
                    self.prior_sx_attr_2[self.sx2idx[sx]] = self.SX_ATTR_NEG_IDX
            for sx in self.sx2idx.keys():
                if self.sx2idx[sx] not in self.prior_sx_attr.keys() and self.sx2idx[sx] not in self.prior_sx_attr_2.keys():
                    self.no_key_sx.append(self.sx2idx[sx])
        # print(len(self.prior_sx_attr))
        # print(len(self.prior_sx_attr_2))
        # print((self.no_key_sx))
        # assert 0
        assert self.num_sxs == len(self.sx2idx) == len(self.idx2sx)

    def add_symptom(self, sx: str) -> None:
        if sx not in self.sx2idx:
            self.sx2idx[sx] = self.num_sxs
            self.sx2count[sx] = 1
            self.idx2sx[self.num_sxs] = sx
            self.num_sxs += 1
        else:
            self.sx2count[sx] += 1

    def update_voc(self, samples: list) -> int:
        for sample in samples:
            for sx in sample['exp_sxs']:
                self.add_symptom(sx)
            for sx in sample['imp_sxs']:
                self.add_symptom(sx)
        return len(samples)

    def trim_voc(self):
        sxs = [sx for sx in sorted(self.sx2count, key=self.sx2count.get, reverse=True)]
        if self.min_sx_freq is not None:
            sxs = [sx for sx in sxs if self.sx2count.get(sx) >= self.min_sx_freq]
        if self.max_voc_size is not None:
            sxs = sxs[: self.max_voc_size]
        sxs = self.special_sxs + sxs
        self.sx2idx = {sx: idx for idx, sx in enumerate(sxs)}
        self.idx2sx = {idx: sx for idx, sx in enumerate(sxs)}
        self.sx2count = {sx: self.sx2count.get(sx) for sx in sxs if sx in self.sx2count}
        self.num_sxs = len(self.sx2idx)
        print('trimmed to {} symptoms with {} special symptoms'.
              format(self.num_sxs - self.num_special, self.num_special))

    def encode(self, sxs: dict, keep_unk=True, add_start=False, add_end=False):
        sx_ids, attr_ids = [], []
        if add_start:
            sx_ids.append(self.start_idx)
            attr_ids.append(self.SX_ATTR_PAD_IDX)
        for sx, attr in sxs.items():
            if sx in self.sx2idx:
                sx_ids.append(self.sx2idx.get(sx))
                attr_ids.append(self.SX_ATTR_MAP.get(attr))
            else:
                if keep_unk:
                    sx_ids.append(self.unk_idx)
                    attr_ids.append(self.SX_ATTR_MAP.get(attr))
        if add_end:
            sx_ids.append(self.end_idx)
            attr_ids.append(self.SX_ATTR_PAD_IDX)
        return sx_ids, attr_ids

    def decoder(self, sx_ids, attr_ids):
        sx_attr = {}
        for sx_id, attr_id in zip(sx_ids, attr_ids):
            if attr_id not in [self.SX_ATTR_PAD_IDX, self.SX_ATTR_NM_IDX]:
                sx_attr.update({self.idx2sx.get(sx_id): self.SX_ATTR_MAP_INV.get(attr_id)})
        return sx_attr

    def __len__(self) -> int:
        return self.num_sxs

    @property
    def pad_idx(self) -> int:
        return self.sx2idx.get(self.SX_PAD)

    @property
    def start_idx(self) -> int:
        return self.sx2idx.get(self.SX_START)

    @property
    def end_idx(self) -> int:
        return self.sx2idx.get(self.SX_END)

    @property
    def unk_idx(self) -> int:
        return self.sx2idx.get(self.SX_UNK)

    @property
    def cls_idx(self) -> int:
        return self.sx2idx.get(self.SX_CLS)

    @property
    def mask_idx(self) -> int:
        return self.sx2idx.get(self.SX_MASK)

    @property
    def pad_sx(self) -> str:
        return self.SX_PAD

    @property
    def start_sx(self) -> str:
        return self.SX_START

    @property
    def end_sx(self) -> str:
        return self.SX_END

    @property
    def unk_sx(self) -> str:
        return self.SX_UNK

    @property
    def cls_sx(self) -> str:
        return self.SX_CLS

    @property
    def mask_sx(self) -> str:
        return self.SX_MASK


class DiseaseVocab:

    def __init__(self, samples: list = None):

        # dis is short for disease
        self.dis2idx = {}
        self.idx2dis = {}
        self.dis2count = {}
        self.num_dis = 0

        # update vocabulary
        if samples is not None:
            if not isinstance(samples, tuple):
                samples = (samples,)
            num_samples = 0
            for split in samples:
                num_samples += self.update_voc(split)
            print('disease vocabulary constructed using {} split and {} samples\nnum of unique diseases: {}'.
                  format(len(samples), num_samples, self.num_dis))

    def add_disease(self, dis: str) -> None:
        if dis not in self.dis2idx:
            self.dis2idx[dis] = self.num_dis
            self.dis2count[dis] = 1
            self.idx2dis[self.num_dis] = dis
            self.num_dis += 1
        else:
            self.dis2count[dis] += 1

    def update_voc(self, samples: list) -> int:
        for sample in samples:
            self.add_disease(sample['label'])
        return len(samples)

    def __len__(self) -> int:
        return self.num_dis

    def encode(self, dis):
        return self.dis2idx.get(dis)


class SymptomDataset(Dataset):

    def __init__(self, samples, sv: SymptomVocab, dv: DiseaseVocab, keep_unk: bool,
                 add_src_start: bool = False, add_tgt_start: bool = False, add_tgt_end: bool = False, train_mode: bool = False):
        self.samples = samples
        self.sv = sv
        self.dv = dv
        self.keep_unk = keep_unk
        self.size = len(self.sv)
        self.add_src_start = add_src_start
        self.add_tgt_start = add_tgt_start
        self.add_tgt_end = add_tgt_end
        self.train_mode = train_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        exp_sx_ids, exp_attr_ids = self.sv.encode(
            sample['exp_sxs'], keep_unk=self.keep_unk, add_start=self.add_src_start)
        imp_sx_ids, imp_attr_ids = self.sv.encode(
            sample['imp_sxs'], keep_unk=self.keep_unk, add_start=self.add_tgt_start, add_end=self.add_tgt_end)
        exp_exe_ids = [0 for i in range(len(exp_sx_ids))]
        imp_exe_ids = [0]
        imp_exe_ids.extend([0 for i in range(len(imp_sx_ids) - 2)])
        imp_exe_ids.append(0)
        key_imp_sx_ids, _ = self.sv.encode(
            sample['imp_key_sxs'], keep_unk=self.keep_unk, add_start=False, add_end=False)
        
        exp_sx_ids, exp_attr_ids, exp_exe_ids, imp_sx_ids, imp_attr_ids, imp_exe_ids, label, key_imp_sx_ids = to_tensor_vla(
            exp_sx_ids, exp_attr_ids, exp_exe_ids, imp_sx_ids, imp_attr_ids, imp_exe_ids, self.dv.encode(sample['label']), key_imp_sx_ids, dtype=torch.long)
        item = {
            'exp_sx_ids': exp_sx_ids,
            'exp_attr_ids': exp_attr_ids,
            'exp_exe_ids': exp_exe_ids,
            'imp_sx_ids': imp_sx_ids,
            'imp_attr_ids': imp_attr_ids,
            'imp_exe_ids': imp_exe_ids,
            'label': label,
            'key_imp_sx_ids': key_imp_sx_ids
        }
        return item


# language model
def lm_collater(samples):
    sx_ids = pad_sequence(
        [torch.cat([sample['exp_sx_ids'], sample['imp_sx_ids']]) for sample in samples], padding_value=0)
    attr_ids = pad_sequence(
        [torch.cat([sample['exp_attr_ids'], sample['imp_attr_ids']]) for sample in samples], padding_value=0)
    exe_ids = pad_sequence(
        [torch.cat([sample['exp_exe_ids'], sample['imp_exe_ids']]) for sample in samples], padding_value=0)
    labels = torch.stack([sample['label'] for sample in samples])
    items = {
        'sx_ids': sx_ids,
        'attr_ids': attr_ids,
        'exe_ids': exe_ids,
        'labels': labels
    }
    return items


# policy gradient
def pg_collater(samples):
    exp_sx_ids = pad_sequence([sample['exp_sx_ids'] for sample in samples], padding_value=0)
    exp_attr_ids = pad_sequence([sample['exp_attr_ids'] for sample in samples], padding_value=0)
    exp_exe_ids = pad_sequence([sample['exp_exe_ids'] for sample in samples], padding_value=0)
    imp_sx_ids = pad_sequence([sample['imp_sx_ids'] for sample in samples], padding_value=0)
    imp_attr_ids = pad_sequence([sample['imp_attr_ids'] for sample in samples], padding_value=0)
    imp_exe_ids = pad_sequence([sample['imp_exe_ids'] for sample in samples], padding_value=0)
    labels = torch.stack([sample['label'] for sample in samples])
    key_imp_sx_ids = pad_sequence([sample['key_imp_sx_ids'] for sample in samples], padding_value=0)
    items = {
        'exp_sx_ids': exp_sx_ids,
        'exp_attr_ids': exp_attr_ids,
        'exp_exe_ids': exp_exe_ids,
        'imp_sx_ids': imp_sx_ids,
        'imp_attr_ids': imp_attr_ids,
        'imp_exe_ids': imp_exe_ids, 
        'labels': labels,
        'key_imp_sx_ids': key_imp_sx_ids
    }
    return items



class PatientSimulator:

    def __init__(self, sv: SymptomVocab):
        self.sv = sv

    def init_sx_ids(self, bsz):
        return torch.full((1, bsz), self.sv.start_idx, dtype=torch.long, device=device)

    def init_attr_ids(self, bsz):
        return torch.full((1, bsz), self.sv.SX_ATTR_PAD_IDX, dtype=torch.long, device=device)

    def answer(self, action, batch):
        d_action, attr_ids = [], []
        for idx, act in enumerate(action):
            if act.item() < self.sv.num_special:
                attr_ids.append(self.sv.SX_ATTR_PAD_IDX)
                d_action.append(self.sv.pad_idx)
            else:
                indices = batch['imp_sx_ids'][:, idx].eq(act.item()).nonzero(as_tuple=False)
                if len(indices) > 0:
                    attr_ids.append(batch['imp_attr_ids'][indices[0].item(), idx].item())
                    d_action.append(act)
                else:
                    attr_ids.append(self.sv.SX_ATTR_NM_IDX)
                    d_action.append(self.sv.pad_idx)
        return to_tensor_vla(d_action, attr_ids)


# symptom inquiry epoch recorder
class SIRecorder:

    def __init__(self, num_samples, num_imp_sxs, digits):
        self.epoch_rewards = defaultdict(list)  # 模拟的每一个序列的每一个查询的症状的奖励
        self.epoch_num_turns = defaultdict(list)  # 模拟的每一个序列的询问的轮数

        self.epoch_num_hits = defaultdict(list)  # 模拟的每一个序列的症状命中总个数
        self.epoch_num_pos_hits = defaultdict(list)  # 模拟的每一个序列的症状（yes）命中个数
        self.epoch_num_neg_hits = defaultdict(list)  # 模拟的每一个序列的症状（no）命中个数
        self.epoch_num_ns_hits = defaultdict(list)  # 模拟的每一个序列的症状（not sure）命中个数

        self.epoch_num_repeats = defaultdict(list)  # 模拟的每一个序列的询问的重复症状的个数
        self.epoch_distances = defaultdict(list)  # 模拟的每一个序列的杰卡德距离（不考虑顺序）
        self.epoch_bleus = defaultdict(list)  # 模拟的每一个序列的BLEU（考虑顺序）

        self.epoch_valid_sx = defaultdict(list)  # 模拟的每一个序列的各轮次有效症状
        self.epoch_all_sx = defaultdict(list)  # 模拟的每一个序列的各轮次总症状
        self.epoch_key_sx = defaultdict(list)  # 模拟的每一个序列的各轮次关键症状

        self.epoch_he = defaultdict(list)  # 模拟的每一个序列的各轮次关键症状

        self.epoch_ai_all_sx = defaultdict(list)  # 模拟的每一个序列的各轮次总症状(ai负责)
        self.epoch_ai_key_sx = defaultdict(list)  # 模拟的每一个序列的各轮次关键症状（ai负责）

        self.epoch_doc_ai_all_sx = defaultdict(list)  # 模拟的每一个序列的各轮次总症状(ai负责在doc的首次)
        self.epoch_doc_ai_key_sx = defaultdict(list)  # 模拟的每一个序列的各轮次关键症状（ai负责在doc的首次）


        self.num_samples = num_samples


        num_pos_imp_sxs, num_neg_imp_sxs, num_ns_imp_sxs, num_key_sxs = num_imp_sxs
        self.num_pos_imp_sxs = num_pos_imp_sxs
        self.num_neg_imp_sxs = num_neg_imp_sxs
        self.num_ns_imp_sxs = num_ns_imp_sxs
        self.num_key_sxs = num_key_sxs
        self.num_imp_sxs = sum((num_pos_imp_sxs, num_neg_imp_sxs, num_ns_imp_sxs))
        self.digits = digits

        self.epoch_acc = defaultdict(float)

    def update(self, batch_rewards, batch_num_turns, batch_num_hits, batch_num_pos_hits,
               batch_num_neg_hits, batch_num_ns_hits, batch_num_repeats, batch_distances, 
               batch_bleus, epoch, he_value, batch_valid_sx, batch_all_sx, batch_key_sx,
               batch_ai_all_sx, batch_ai_key_sx,batch_doc_ai_all_sx,batch_doc_ai_key_sx):
        self.epoch_rewards[epoch].extend(batch_rewards)
        self.epoch_num_turns[epoch].extend(batch_num_turns)

        self.epoch_num_hits[epoch].extend(batch_num_hits)
        self.epoch_num_pos_hits[epoch].extend(batch_num_pos_hits)
        self.epoch_num_neg_hits[epoch].extend(batch_num_neg_hits)
        self.epoch_num_ns_hits[epoch].extend(batch_num_ns_hits)

        self.epoch_num_repeats[epoch].extend(batch_num_repeats)
        self.epoch_distances[epoch].extend(batch_distances)
        self.epoch_bleus[epoch].extend(batch_bleus)
        
        self.epoch_valid_sx[epoch].extend(batch_valid_sx)
        self.epoch_all_sx[epoch].extend(batch_all_sx)
        self.epoch_key_sx[epoch].extend(batch_key_sx)

        self.epoch_he[epoch].extend(he_value)

        self.epoch_ai_all_sx[epoch].extend(batch_ai_all_sx)
        self.epoch_ai_key_sx[epoch].extend(batch_ai_key_sx)

        self.epoch_doc_ai_all_sx[epoch].extend(batch_doc_ai_all_sx)
        self.epoch_doc_ai_key_sx[epoch].extend(batch_doc_ai_key_sx)

    def update_acc(self, epoch, acc):
        self.epoch_acc[epoch] = acc

    def epoch_summary(self, epoch):
        avg_epoch_rewards = average(self.epoch_rewards[epoch], self.num_imp_sxs)
        avg_epoch_turns = average(self.epoch_num_turns[epoch], self.num_samples)

        avg_epoch_num_hits = average(self.epoch_num_hits[epoch], self.num_imp_sxs)
        avg_epoch_num_pos_hits = average(self.epoch_num_pos_hits[epoch], self.num_pos_imp_sxs)
        avg_epoch_num_neg_hits = average(self.epoch_num_neg_hits[epoch], self.num_neg_imp_sxs)
        avg_epoch_num_ns_hits = average(self.epoch_num_ns_hits[epoch], self.num_ns_imp_sxs)

        avg_epoch_num_repeats = average(self.epoch_num_repeats[epoch], self.num_samples)
        avg_epoch_distances = average(self.epoch_distances[epoch], self.num_samples)
        avg_epoch_bleus = average(self.epoch_bleus[epoch], self.num_samples)

        sx_acc = compute_sx_acc(self.epoch_valid_sx[epoch], self.epoch_all_sx[epoch])
        avg_sx_acc = average(self.epoch_valid_sx[epoch], self.epoch_all_sx[epoch])
        
        key_sx_acc = compute_sx_acc(self.epoch_key_sx[epoch], self.epoch_all_sx[epoch])
        avg_key_sx_acc = average(self.epoch_key_sx[epoch], self.epoch_all_sx[epoch])

        key_sx_rec = average(self.epoch_key_sx[epoch], self.num_key_sxs)

        epoch_acc = self.epoch_acc[epoch] if epoch in self.epoch_acc else 0

        he_value = average(self.epoch_he[epoch], self.num_samples)

        avg_ai_key_sx_acc = average(self.epoch_ai_key_sx[epoch], self.epoch_ai_all_sx[epoch])

        avg_doc_ai_key_sx_acc = average(self.epoch_doc_ai_key_sx[epoch], self.epoch_doc_ai_all_sx[epoch])

        print(
            'epoch: {} -> rewards: {}, all/pos/neg/ns hits: {}/{}/{}/{}, acc: {}, turns: {}, HE: {}\nsx_acc: {} -> {}'.
            format(epoch + 1,
                   round(avg_epoch_rewards, self.digits),
                   round(avg_epoch_num_hits, self.digits),
                   round(avg_epoch_num_pos_hits, self.digits),
                   round(avg_epoch_num_neg_hits, self.digits),
                   round(avg_epoch_num_ns_hits, self.digits),
                   round(epoch_acc, self.digits),
                   round(avg_epoch_turns, self.digits),
                   round(he_value, self.digits),
                   round(avg_ai_key_sx_acc, self.digits),
                   sx_acc))

    @staticmethod
    def lmax(arrays: list):
        cur_val = arrays[-1]
        max_val = max(arrays)
        max_index = len(arrays) - arrays[::-1].index(max_val) - 1
        return cur_val, max_val, max_index
    
    @staticmethod
    def lmax_acc(arrays: list, hes: list):
        cur_val = arrays[-1]
        max_val = max(arrays)
        max_indexs = []
        for indx in range(len(arrays)):
            if arrays[indx] == max_val:
                max_indexs.append(indx)
        min_he = 10
        for inx in max_indexs:
            if min_he > hes[inx]:
                max_index = inx
                min_he = hes[inx]
        return cur_val, max_val, max_index

    def report(self, max_epoch: int, digits: int, alpha: float = 0.2, verbose: bool = False):
        
        recs = [average(self.epoch_num_hits[epoch], self.num_imp_sxs) for epoch in range(max_epoch + 1)]
        HEs = [average(self.epoch_he[epoch], self.num_samples) for epoch in range(max_epoch + 1)]
        turns = [average(self.epoch_num_turns[epoch], self.num_samples) for epoch in range(max_epoch + 1)]
        cur_rec, best_rec, best_rec_epoch = self.lmax(recs)
        accs = [self.epoch_acc[epoch] for epoch in range(max_epoch + 1)]
        cur_acc, best_acc, best_acc_epoch = self.lmax_acc(accs, HEs)
        mets = [alpha * rec + (1 - alpha) * acc for rec, acc in zip(recs, accs)]
        cur_met, best_met, best_met_epoch = self.lmax(mets)
        best_rec_acc, best_acc_rec, best_met_rec, best_met_acc = \
            accs[best_rec_epoch], recs[best_acc_epoch], recs[best_met_epoch], accs[best_met_epoch]
        best_acc_HE, best_acc_turn = HEs[best_acc_epoch], turns[best_acc_epoch]
        if verbose:
            print('best recall -> epoch: {}, recall: {}, accuracy: {}\nbest accuracy -> epoch: {}, he: {}, turn: {}, accuracy: {}\nbest metric -> epoch: {}, recall: {}, accuracy: {}'.format(
                best_rec_epoch + 1, round(best_rec, digits), round(best_rec_acc, digits), 
                best_acc_epoch + 1, round(best_acc_HE, digits), round(best_acc_turn, digits), round(best_acc, digits),
                best_met_epoch + 1, round(best_met_rec, digits), round(best_met_acc, digits)
            ))
        return cur_rec, best_rec, best_rec_epoch, best_rec_acc, cur_acc, best_acc, best_acc_epoch, best_acc_rec, cur_met, best_met, best_met_epoch, best_met_rec, best_met_acc, best_acc_HE, best_acc_turn


def compute_sx_acc(valid, all):
    val_sx = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    all_sx = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for id in range(len(valid)):
        val_sx[id % num_turns] += valid[id]
        all_sx[id % num_turns] += all[id]
    return [ round(x/y, 3) if y!=0 else 0 for x,y in zip(val_sx, all_sx) ]

def recursive_sum(item):
    if isinstance(item, list):
        try:
            return sum(item)
        except TypeError:
            return recursive_sum(sum(item, []))
    else:
        return item


def average(numerator, denominator):
    return 0 if recursive_sum(denominator) == 0 else recursive_sum(numerator) / recursive_sum(denominator)


def to_numpy(tensors):
    arrays = {}
    for key, tensor in tensors.items():
        arrays[key] = tensor.cpu().numpy()
    return arrays


def to_numpy_(tensor):
    return tensor.cpu().numpy()


def to_list(tensor):
    return to_numpy_(tensor).tolist()


def to_numpy_vla(*tensors):
    arrays = []
    for tensor in tensors:
        arrays.append(to_numpy_(tensor))
    return arrays


def to_tensor_(array, dtype=None):
    if dtype is None:
        return torch.tensor(array, device=device)
    else:
        return torch.tensor(array, dtype=dtype, device=device)


def to_tensor_vla(*arrays, dtype=None):
    tensors = []
    for array in arrays:
        tensors.append(to_tensor_(array, dtype))
    return tensors


def compute_num_sxs(samples, sv: SymptomVocab):
    num_yes_imp_sxs = 0
    num_no_imp_sxs = 0
    num_not_sure_imp_sxs = 0
    num_key_sxs = 0
    for sample in samples:
        for sx, attr in sample['imp_sxs'].items():
            if attr == '0':
                num_no_imp_sxs += 1
                if sx in sv.sx2idx.keys() and sv.sx2idx[sx] in sv.prior_sx_attr_2.keys():
                    num_key_sxs += 1
            elif attr == '1':
                num_yes_imp_sxs += 1
                if sx in sv.sx2idx.keys() and sv.sx2idx[sx] in sv.prior_sx_attr.keys():
                    num_key_sxs += 1
            else:
                num_not_sure_imp_sxs += 1
    return num_yes_imp_sxs, num_no_imp_sxs, num_not_sure_imp_sxs, num_key_sxs


def combinations(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


# compute co-occurrence matrix
def compute_dscom(samples, sv: SymptomVocab, dv: DiseaseVocab):
    cm = np.zeros((dv.num_dis, sv.num_sxs))
    if not isinstance(samples, tuple):
        samples = (samples,)
    for split in samples:
        for sample in split:
            exp_sx_ids, _ = sv.encode(sample['exp_sxs'])
            imp_sx_ids, _ = sv.encode(sample['imp_sxs'])
            dis_id = dv.encode(sample['label'])
            for sx_id in exp_sx_ids + imp_sx_ids:
                cm[dis_id][sx_id] += 1
    return cm


def compute_sscom(samples, sv: SymptomVocab, smooth: bool = True, normalize: bool = True):
    cm = np.zeros((sv.num_sxs, sv.num_sxs))
    for sample in samples:
        exp_sx_ids, _ = sv.encode(sample['exp_sxs'])
        imp_sx_ids, _ = sv.encode(sample['imp_sxs'])
        sxs = exp_sx_ids + imp_sx_ids
        for (i, j) in combinations(sxs, 2):
            cm[i][j] += 1
            cm[j][i] += 1
    if smooth:
        min_val, max_val = .1, .5
        for i in range(sv.num_sxs):
            for j in range(sv.num_sxs):
                if cm[i][j] == 0:
                    cm[i][j] = truncnorm.rvs(min_val, max_val, size=1)[0]
    if normalize:
        cm = cm / cm.sum(axis=1).reshape(-1, 1)
    return cm


def random_agent(samples, sv: SymptomVocab, max_turn: int, exclude_exp: bool = True, times: int = 100):
    recs = []
    for _ in range(times):
        num_imp_sxs, num_hits = 0, 0
        for sample in samples:
            exp_sx_ids, _ = sv.encode(sample['exp_sxs'], keep_unk=False)
            imp_sx_ids, _ = sv.encode(sample['imp_sxs'], keep_unk=False)
            if exclude_exp:
                action_space = [sx_id for sx_id, _ in sv.idx2sx.items() if sx_id not in exp_sx_ids]
            else:
                action_space = [sx_id for sx_id, _ in sv.idx2sx.items()]
            actions = np.random.choice(action_space, size=max_turn, replace=False)
            num_imp_sxs += len(imp_sx_ids)
            num_hits += len([action for action in actions if action in imp_sx_ids])
        recs.append(num_hits / num_imp_sxs)
    return recs


def rule_agent(samples, cm, sv: SymptomVocab, max_turn: int, exclude_exp: bool = True):
    num_imp_sxs, num_pos_imp_sxs, num_neg_imp_sxs = 0, 0, 0
    num_hits, num_pos_hits, num_neg_hits = 0, 0, 0
    for sample in samples:
        exp_sx_ids, _ = sv.encode(sample['exp_sxs'], keep_unk=False)
        imp_sx_ids, imp_attr_ids = sv.encode(sample['imp_sxs'], keep_unk=False)
        imp_pos_sx_ids = [sx_id for sx_id, attr_id in zip(imp_sx_ids, imp_attr_ids) if attr_id == sv.SX_ATTR_POS_IDX]
        imp_neg_sx_ids = [sx_id for sx_id, attr_id in zip(imp_sx_ids, imp_attr_ids) if attr_id == sv.SX_ATTR_NEG_IDX]
        num_imp_sxs += len(imp_sx_ids)
        num_pos_imp_sxs += len(imp_pos_sx_ids)
        num_neg_imp_sxs += len(imp_neg_sx_ids)
        actions = []
        current = set(exp_sx_ids)
        previous = set()
        for step in range(max_turn):
            # similarity score
            sim = np.zeros(sv.num_sxs)
            for sx in current:
                sim += cm[sx]
            index = -1
            if exclude_exp:
                for index in np.flip(np.argsort(sim)):
                    if index not in current.union(previous):
                        break
            else:
                for index in np.flip(np.argsort(sim)):
                    if index not in previous:
                        break
            # if index in imp_sx_ids and imp_attr_ids[imp_sx_ids.index(index)] == sv.SX_ATTR_POS_IDX:
            #     current.add(index)
            # if index in imp_sx_ids:
            #     current.add(index)
            previous.add(index)
            actions.append(index)
        num_hits += len([sx_id for sx_id in actions if sx_id in imp_sx_ids])
        num_pos_hits += len([sx_id for sx_id in actions if sx_id in imp_pos_sx_ids])
        num_neg_hits += len([sx_id for sx_id in actions if sx_id in imp_neg_sx_ids])
    rec = num_hits / num_imp_sxs
    pos_rec = num_pos_hits / num_pos_imp_sxs
    neg_rec = num_neg_hits / num_neg_imp_sxs
    return rec, pos_rec, neg_rec


class RewardDistributor:

    def __init__(self, sv: SymptomVocab, dv: DiseaseVocab, dscom):

        self.sv = sv
        self.dv = dv

        # 症状恢复奖励
        self.pos_priori_reward = 1.0
        self.neg_priori_reward = -1.0
        self.hit_reward = {1: 5.0, 2: 3.0, 3: 0.1}
        self.decline_rate = 0.0
        self.repeat_reward = 0.0
        self.end_reward = 0.0
        self.missed_reward = -0.2
        self.zero_reward = 0.0

        self.dscom = dscom

    def compute_sr_priori_reward(self, action, dis_id, eps=0.0):
        # 先验奖励，push智能体不生成无关的症状（由语料库中的疾病-症状共现矩阵决定）
        reward = []
        for act in action:
            if self.dscom[dis_id, act] > eps:
                reward.append(self.zero_reward)
            else:
                reward.append(self.zero_reward)
        return reward

    def compute_sr_ground_reward(self, action, imp_sx, imp_attr, num_hit, suc_diag):
        # 真实奖励，push智能体生成ground truth中的症状
        # 1. 如果智能体生成了隐形症状中包含的关键症状，给予正向奖励
        # 2. 如果智能体生成了隐形症状中包含的症状，给予0奖励
        # 3. 如果智能体生成了隐形症状中不包含的症状，给予负向奖励
        reward = []
        history_acts, num_repeat = set(), 0
        for i, act in enumerate(action):
            if act in history_acts: #不会出现
                num_repeat += 1
                reward.append(self.repeat_reward)
            else:
                history_acts.add(act)
                if act == self.sv.end_idx: #不会出现
                    reward.append(num_hit - len(imp_sx) + self.end_reward)
                else:
                    if suc_diag == True:
                        diag_reward = self.hit_reward[1]
                    else:
                        diag_reward = self.zero_reward
                    if act in imp_sx:
                        if train_dataset != 'mz10':
                            reward.append(self.hit_reward[3] + diag_reward)
                        else:
                            reward.append(self.hit_reward[1])
                    else:
                        reward.append(self.missed_reward)
        return reward, num_repeat

    @staticmethod
    def compute_sr_global_reward(action, imp_sx, num_hit, eps=1e-3):
        # 全局奖励，push智能体生成与真实情况下，顺序尽可能类似的序列
        # 1.非序列相关奖励（杰卡德距离）
        set(action).intersection()
        distance = (num_hit + eps) / (len(set(action).union(set(imp_sx))) + eps)
        # 2.序列相关奖励（BLEU，其中denoise action是action和隐形症状序列中的公共子序列）
        denoise_action = [act for act in action if act in imp_sx]
        bleu = sentence_bleu([imp_sx], denoise_action, smoothing_function=SmoothingFunction().method1)
        # 这些奖励仅分配到命中的那些症状（）
        distance = [0 if act in imp_sx else 0 for act in action]
        bleu = [0 if act in imp_sx else 0 for act in action]
        return distance, bleu, denoise_action

    # 计算症状恢复奖励（symptom recovery）
    def compute_sr_reward(self, actions, np_batch, epoch, he_value, act_role, sir: SIRecorder, diag_hits=None, early_stop=None):
        batch_size, seq_len = actions.shape
        # 将 actions 转化为 numpy array
        actions = to_numpy_(actions)
        
        # 初始化奖励，询问轮数，症状命中数（yes/no/not sure）
        rewards, num_turns, num_hits, num_pos_hits, num_neg_hits, num_ns_hits = [], [], [], [], [], []
        # 初始化重复次数，去噪的动作序列
        num_repeats, denoise_actions, distances, bleus = [], [], [], []
        # 每轮关键症状数，有效症状数，总数
        key_sx = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        valid_sx = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        all_sx = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # 机器做出动作的准确率
        ai_key_sx = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ai_all_sx = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # after doc ai
        doc_ai_key_sx = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        doc_ai_all_sx = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        # print(act_role)
        act_all_id = 0
        he_val = 0
        # assert 0
        # 计算每一个序列的回报
        for idx in range(batch_size):
            if early_stop is not None:
                early_stop_seq = to_numpy_(early_stop)
            # break
            # 得到隐形症状序列的yes/no/not sure子序列
            imp_sx, imp_attr, imp_pos_sx, imp_neg_sx, imp_ns_sx = self.truncate_imp_sx(idx, np_batch)
            # 得到动作序列（通过结束症状id进行截断）
            action = self.truncate_action(idx, actions)
            if early_stop is not None:
                early_stop_seq = to_numpy_(early_stop[act_all_id: act_all_id+ len(action)])
                act_all_id += len(action)
                action = self.truncate_action_early(early_stop_seq, action)
                
            if diag_hits is not None:
                suc_diag = diag_hits[idx].item()
            else:
                suc_diag = True

            action_role = act_role[idx]
            last_human = False

            # print(action_role)
            # assert 0
            # 记录每次问症状的准确率
            if len(action) != 0:
                for vi in range(len(action)):
                    all_sx[vi] = all_sx[vi] + 1
                    if action[vi] in imp_sx:
                        valid_sx[vi] = valid_sx[vi] + 1
                    if action_role[vi] == False:
                        he_val += 1
                    # ai询问症状
                    if action_role[vi] == True:
                        ai_all_sx[vi] = ai_all_sx[vi] + 1
                    if last_human == True:
                        doc_ai_all_sx[vi] = doc_ai_all_sx[vi] + 1
                    
                    if vi >= human_step_s and vi <= human_step and action_role[vi] == False:
                        last_human = True
            # 计算询问轮数，症状命中数等
            num_turns.append(len(action))
            num_hit = len(set(action).intersection(set(imp_sx)))
            num_hits.append(num_hit)
            num_pos_hits.append(len(set(action).intersection(set(imp_pos_sx))))
            num_neg_hits.append(len(set(action).intersection(set(imp_neg_sx))))
            num_ns_hits.append(len(set(action).intersection(set(imp_ns_sx))))
            # 计算先验奖励
            priori_reward = self.compute_sr_priori_reward(action, dis_id=np_batch['labels'][idx])
            # 计算真实奖励
            ground_reward, num_repeat = self.compute_sr_ground_reward(action, imp_sx, imp_attr, num_hit, suc_diag)
            last_human = False
            if len(ground_reward) != 0:
                for vi in range(len(ground_reward)):
                    if ground_reward[vi] > 0:
                        key_sx[vi] = key_sx[vi] + 1
                        # ai询问关键症状
                        if action_role[vi] == True:
                            ai_key_sx[vi] = ai_key_sx[vi] + 1
                            
                            # last_human = False
                    if last_human == True:
                        if ground_reward[vi] > 0:
                            doc_ai_key_sx[vi] = doc_ai_key_sx[vi] + 1
                        
                    if vi >= human_step_s and vi <= human_step and action_role[vi] == False:
                        last_human = True
            num_repeats.append(num_repeat)
            # 计算全局奖励
            distance, bleu, denoise_action = self.compute_sr_global_reward(action, imp_sx, num_hit)
            distances.append(0 if len(distance) == 0 else max(distance))
            bleus.append(0 if len(bleu) == 0 else max(bleu))
            denoise_actions.append(denoise_action)
            # 计算最终奖励（每一步的奖励）
            reward = [pr + gr + dr + br for pr, gr, dr, br in zip(priori_reward, ground_reward, distance, bleu)]
            reward += [0] * (seq_len - len(action))
            rewards.append(reward)
        he_value = []
        he_value.append(he_val)
        
        sir.update(rewards, num_turns, num_hits, num_pos_hits, num_neg_hits,
                   num_ns_hits, num_repeats, distances, bleus, epoch=epoch, he_value = he_value, 
                   batch_valid_sx = valid_sx, batch_all_sx = all_sx, batch_key_sx = key_sx, 
                   batch_ai_all_sx = ai_all_sx, batch_ai_key_sx = ai_key_sx,
                   batch_doc_ai_all_sx = doc_ai_all_sx, batch_doc_ai_key_sx = doc_ai_key_sx)
        return rewards

    def truncate_action(self, idx: int, actions: list) -> list:
        action = actions[idx].tolist()
        return action[: action.index(self.sv.end_idx)] if self.sv.end_idx in action else action

    def truncate_action_early(self, early_stop, action) -> list:
        for i in range(len(early_stop)):
            if early_stop[i] > disease_conf:
                action[i] = self.sv.end_idx
                break
        return action[: action.index(self.sv.end_idx)] if self.sv.end_idx in action else action
    def truncate_imp_sx(self, idx: int, np_batch: dict):
        imp_sx, imp_attr, imp_pos_sx, imp_neg_sx, imp_ns_sx = [], [], [], [], []
        for sx_id, attr_id in zip(np_batch['imp_sx_ids'][1:, idx], np_batch['imp_attr_ids'][1:, idx]):
            if sx_id == self.sv.end_idx:
                break
            else:
                imp_sx.append(sx_id)
                imp_attr.append(attr_id)
                if attr_id == self.sv.SX_ATTR_POS_IDX:
                    imp_pos_sx.append(sx_id)
                elif attr_id == self.sv.SX_ATTR_NEG_IDX:
                    imp_neg_sx.append(sx_id)
                elif attr_id == self.sv.SX_ATTR_NS_IDX:
                    imp_ns_sx.append(sx_id)
        return imp_sx, imp_attr, imp_pos_sx, imp_neg_sx, imp_ns_sx

def compute_value(actions, values, ):
    actions = to_numpy_(actions)
    bz, seq_len = values.shape
    masks = []
    # 计算每一个序列的v值
    for idx in range(bz):
        action = actions[idx].tolist()
        if 2 in action:
            action = action[: action.index(2)]
        mask = [1 for i in action]
        mask += [0] * (seq_len - len(action))
        masks.append(mask)
    values = values * to_tensor_(masks)
    return values

def make_features_neural(sx_ids, attr_ids, labels, sv: SymptomVocab):
    from conf import suffix
    feats = []
    for sx_id, attr_id in zip(sx_ids, attr_ids):
        feature = []
        sample = sv.decoder(sx_id, attr_id)
        for sx, attr in sample.items():
            feature.append(sx + suffix.get(attr))
        feats.append(' '.join(feature))
    return feats, labels


def extract_features(sx_ids, attr_ids, sv: SymptomVocab):
    sx_feats, attr_feats = [], []
    exe_feats = []
    for idx in range(len(sx_ids)):
        sx_feat, attr_feat, exe_feat = [sv.start_idx], [sv.SX_ATTR_PAD_IDX], [sv.SX_ATTR_PAD_IDX]
        for sx_id, attr_id in zip(sx_ids[idx], attr_ids[idx]):
            if sx_id == sv.end_idx:
                break
            if attr_id not in [sv.SX_ATTR_PAD_IDX, sv.SX_ATTR_NM_IDX]:
                sx_feat.append(sx_id)
                attr_feat.append(attr_id)
                # if train_dataset != 'mz10':
                #     sx_feat.append(sx_id)
                #     attr_feat.append(attr_id)
                # else:
                #         # 去除无效的症状和属性pairs
                #     if sx_id in sv.prior_sx_attr.keys() and attr_id == sv.prior_sx_attr[sx_id]:
                #         # 只保留True的sx(key)
                #         sx_feat.append(sx_id)
                #         attr_feat.append(attr_id)
                #     if sx_id in sv.prior_sx_attr_2.keys() and attr_id == sv.prior_sx_attr_2[sx_id]:
                #         # 只保留False的sx(key)
                #         sx_feat.append(sx_id)
                #         attr_feat.append(attr_id)
                
        sx_feats.append(to_tensor_(sx_feat))
        attr_feats.append(to_tensor_(attr_feat))
        
    return sx_feats, attr_feats



def make_features_xfmr(sv: SymptomVocab, batch, si_sx_ids=None, si_attr_ids=None,  merge_act: bool = False,
                       merge_si: bool = False):
    # convert to numpy
    assert merge_act or merge_si
    sx_feats, attr_feats = [], []
    exe_feats = []
    if merge_act:
        act_sx_ids = torch.cat([batch['exp_sx_ids'], batch['imp_sx_ids']]).permute([1, 0])
        act_attr_ids = torch.cat([batch['exp_attr_ids'], batch['imp_attr_ids']]).permute([1, 0])
        act_sx_ids, act_attr_ids = to_numpy_vla(act_sx_ids, act_attr_ids)
        act_sx_feats, act_attr_feats = extract_features(act_sx_ids, act_attr_ids, sv)
        sx_feats += act_sx_feats
        attr_feats += act_attr_feats
    if merge_si:
        si_sx_ids, si_attr_ids = to_numpy_vla(si_sx_ids, si_attr_ids)
        si_sx_feats, si_attr_feats = extract_features(si_sx_ids, si_attr_ids, sv)
        sx_feats += si_sx_feats
        attr_feats += si_attr_feats
        
    sx_feats = pad_sequence(sx_feats, padding_value=sv.pad_idx).long()
    attr_feats = pad_sequence(attr_feats, padding_value=sv.SX_ATTR_PAD_IDX).long()
    # print(attr_feats)
    # assert 0
    return sx_feats, attr_feats

def extract_features_seq(sx_ids, attr_ids, sv: SymptomVocab):
    sx_feats, attr_feats = [], []
    exe_feats = []
    for idx in range(len(sx_ids)):
        sx_feat, attr_feat, exe_feat = [sv.start_idx], [sv.SX_ATTR_PAD_IDX], [sv.SX_ATTR_PAD_IDX]
        seq_len = len(sx_ids[idx])
        # print(seq_len-num_turns)
        # print(np.where(sx_ids[idx]==1)[0][0])
        # print(sx_ids[idx])
        # assert 0
        for sx_id, attr_id in zip(sx_ids[idx], attr_ids[idx]):
            if sx_id == sv.end_idx:
                break
            if attr_id not in [sv.SX_ATTR_PAD_IDX, sv.SX_ATTR_NM_IDX]:
                sx_feat.append(sx_id)
                attr_feat.append(attr_id)
                # if train_dataset != 'mz10':
                #     sx_feat.append(sx_id)
                #     attr_feat.append(attr_id)
                # else:
                #         # 去除无效的症状和属性pairs
                #     if sx_id in sv.prior_sx_attr.keys() and attr_id == sv.prior_sx_attr[sx_id]:
                #         # 只保留True的sx(key)
                #         sx_feat.append(sx_id)
                #         attr_feat.append(attr_id)
                #     if sx_id in sv.prior_sx_attr_2.keys() and attr_id == sv.prior_sx_attr_2[sx_id]:
                #         # 只保留False的sx(key)
                #         sx_feat.append(sx_id)
                #         attr_feat.append(attr_id)
            if sv.end_idx in sx_ids[idx]:
                seq_end = np.where(sx_ids[idx]==sv.end_idx)[0][0] - 1
            else:
                seq_end = seq_len -1
            if np.where(sx_ids[idx]==sx_id)[0][0] >= seq_len-num_turns-1 and np.where(sx_ids[idx]==sx_id)[0][0] < seq_end:
                sx_feats.append(to_tensor_(sx_feat))
                attr_feats.append(to_tensor_(attr_feat))
                
        # sx_feats.append(to_tensor_(sx_feat))
        # attr_feats.append(to_tensor_(attr_feat))
        
    return sx_feats, attr_feats

def make_features_xfmr_seq(sv: SymptomVocab, batch, si_sx_ids=None, si_attr_ids=None,  merge_act: bool = False,
                       merge_si: bool = False):
    # convert to numpy
    assert merge_act or merge_si
    sx_feats, attr_feats = [], []
    exe_feats = []
    if merge_si:
        si_sx_ids, si_attr_ids = to_numpy_vla(si_sx_ids, si_attr_ids)
        si_sx_feats, si_attr_feats = extract_features_seq(si_sx_ids, si_attr_ids, sv)
        sx_feats += si_sx_feats
        attr_feats += si_attr_feats
    # print(sx_feats)
    sx_feats = pad_sequence(sx_feats, padding_value=sv.pad_idx).long()
    attr_feats = pad_sequence(attr_feats, padding_value=sv.SX_ATTR_PAD_IDX).long()
    # print(attr_feats)
    # assert 0
    return sx_feats, attr_feats

def extract_seq_dec(seq_sx, seq_attr, seq_exe, seq_label, sv: SymptomVocab):
    # seq_sx = torch.tensor([10,  1, 77, 88, 12, 78,40, 79, 8, 61,  2,  0,  0,  0,  0,  0,  0,  0,  0],
    #    device='cuda:0')
    # seq_attr = torch.tensor([1,  0, 1, 1, 1, 1,  1,1,1, 1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    #    device='cuda:0')
    # seq_label = torch.zeros(seq_sx.shape[0], dtype=torch.long, device=device)

    for id in range(len(seq_sx)):
        if seq_sx[id].item() == sv.start_idx:
            start_id = id
        if seq_sx[id].item() == sv.end_idx:
            end_id = id
    key_sx = []
    key_attr = []
    key_id = []
    for idx in range(start_id + 1, end_id):
        if seq_sx[idx].item() in sv.prior_sx_attr.keys() and seq_attr[idx].item() == sv.prior_sx_attr[seq_sx[idx].item()]:
            # 只保留True的sx(key)
            key_sx.append(seq_sx[idx].item())
            key_attr.append(seq_attr[idx].item())
            key_id.append(idx)
        if seq_sx[idx].item() in sv.prior_sx_attr_2.keys() and seq_attr[idx].item() == sv.prior_sx_attr_2[seq_sx[idx].item()]:
            # 只保留False的sx(key)
            key_sx.append(seq_sx[idx].item())
            key_attr.append(seq_attr[idx].item())
            key_id.append(idx)
    for id in range(len(seq_sx)):
        seq_label[id] = seq_sx[id]
        if id in key_id:
            seq_sx[id] = seq_sx[id - 1]
            seq_attr[id] = seq_attr[id - 1]
            seq_exe[id] = seq_exe[id - 1]
    sta_next = 0
    for idx in range(start_id + 1, end_id):
        if len(key_id) == 0:
            break
        sta_next += 1
        if idx < key_id[0]:
            seq_label[idx] = key_sx[0]
        if idx == key_id[0]:
            key_sx.pop(0)
            key_attr.pop(0)
            key_id.pop(0)
    seq_label[start_id + 1 + sta_next] = sv.end_idx
    for id in range(start_id + sta_next + 2, len(seq_sx)):
        seq_label[id] = sv.pad_idx
    # print(seq_sx)
    # print(seq_attr)
    # print(seq_exe)
    # print(seq_label)
    # assert 0
    return seq_sx, seq_attr, seq_exe, seq_label
    
        

def make_pretrain_feat(sx_ids, attr_ids, exe_ids, sv: SymptomVocab):
    sx_ids_labels = torch.zeros((sx_ids.shape[0], sx_ids.shape[1]), dtype=torch.long, device=device).permute(1, 0)
    dec_sx_ids = torch.zeros((sx_ids.shape[0], sx_ids.shape[1]), dtype=torch.long, device=device).permute(1, 0)
    dec_attr_ids = torch.zeros((sx_ids.shape[0], sx_ids.shape[1]), dtype=torch.long, device=device).permute(1, 0)
    dec_exe_ids = torch.zeros((sx_ids.shape[0], sx_ids.shape[1]), dtype=torch.long, device=device).permute(1, 0)


    for seq_id in range(sx_ids.shape[1]):
        for id in range(sx_ids.shape[0]):
            dec_sx_ids[seq_id][id] = sx_ids[id][seq_id]
            dec_attr_ids[seq_id][id] = attr_ids[id][seq_id]
            dec_exe_ids[seq_id][id] = exe_ids[id][seq_id]
           
    
    for seq_id in range(sx_ids.shape[1]):
        dec_sx_ids[seq_id], dec_attr_ids[seq_id], dec_exe_ids[seq_id], sx_ids_labels[seq_id] = extract_seq_dec(dec_sx_ids[seq_id], dec_attr_ids[seq_id], dec_exe_ids[seq_id], sx_ids_labels[seq_id], sv)
        
    return dec_sx_ids.permute(1, 0), dec_attr_ids.permute(1, 0), dec_exe_ids.permute(1, 0), sx_ids_labels.permute(1, 0)

import torch.utils.data as Data

class Memory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
    def push(self,transitions):
        '''_summary_
        Args:
            trainsitions (tuple): _description_
        '''
        self.buffer.append(transitions)
    def sample(self):
        self.torch_dataset = Data.TensorDataset(torch.cat(tuple([a for a,b in self.buffer]),dim = 0), torch.cat(tuple([b for a,b in self.buffer]),dim = 0))
        self.train_loader = Data.DataLoader(dataset=self.torch_dataset, batch_size=128, shuffle=True,drop_last=False)
        self.buffer.clear()
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)
    

class HumanAgent:

    def __init__(self, sv: SymptomVocab, ability):
        self.sv = sv
        self.ability = ability
        self.rect = {}

    def init_batch(self, batch: dict):
        self.batch = batch
    
    def change_ability(self, ability):
        self.ability = ability

    def involve(self, action, sx_ids, attr_ids,action_confs=None):
        sx_ids_list, attr_ids_list = [], []
        mask_log = []
        mask_valid = []
        for idx, act in enumerate(action):
            action_conf = True
            if action_confs is not None:
                if action_confs[idx] >= symptom_conf:
                    action_conf = False
            sx_ids_list = sx_ids[idx].cpu().tolist()
            attr_ids_list = attr_ids[idx].cpu().tolist()
            imp_sx_list = self.batch['imp_sx_ids'].permute([1, 0])[idx].cpu().tolist()
            imp_key_list = self.batch['key_imp_sx_ids'].permute([1, 0])[idx].cpu().tolist()
            if attr_ids_list[-1] == self.sv.SX_ATTR_NM_IDX and attr_ids_list[-2] == self.sv.SX_ATTR_NM_IDX  and\
                self.sv.end_idx not in sx_ids_list and action_conf ==True  : 
                action[idx], val = self.ask(imp_sx_list, sx_ids_list, imp_key_list)
                mask_log.append(0)
                mask_valid.append(val)
            else:
                mask_log.append(1)
                mask_valid.append(1)
        return mask_log, mask_valid
    
    def ask(self, imp_sx_list, sx_ids_list, imp_key_list):
        # print(sx_ids_list)
        # print(imp_sx_list)
        # print(imp_sx_list[1:imp_sx_list.index(self.sv.end_idx)])
        repeat_symptoms = list(set(sx_ids_list).intersection(set(imp_sx_list[1:imp_sx_list.index(self.sv.end_idx)])))
        candidate_symptoms = [x for x in imp_sx_list[1:imp_sx_list.index(self.sv.end_idx)] if x not in repeat_symptoms]
        candidate_key_symptoms = list(set(imp_key_list).intersection(set(candidate_symptoms)))
        if len(candidate_symptoms) == 0:
            return self.sv.end_idx, 1
        greedy = random.random()
        # print(greedy)
        # print(sx_ids_list)
        # print(candidate_symptoms)
        if greedy < self.ability:
            act = random.choice(candidate_symptoms)
            if len(candidate_key_symptoms) != 0 and greedy < key_hp:
                act = random.choice(candidate_key_symptoms)
            if act not in self.rect.keys():
                self.rect[act] = 1
            else:
                self.rect[act] += 1
            return act, 0
        else:
            act = random.choice([x for x in range(self.sv.num_special, self.sv.num_sxs) if x not in sx_ids_list])
            return act, 1

def data_augmentation(samples: list = None, real_prior_feat = None):
    # 增强后的训练数据集
    data_samples = []
    dis2sx = {}
    # 记录各疾病下的症状集
    for sample in samples:
        dis = sample['label']
        if dis not in dis2sx.keys():
            dis2sx[dis] = []
        for k, v in sample['imp_sxs'].items():
            if k not in dis2sx[dis]:
                if v == '1':
                    sy_sta = k + '-True'
                else:
                    sy_sta = k + '-False'
                if sy_sta in real_prior_feat:
                    dis2sx[dis].append(k)
    import copy
    for sample in samples:

        for i in range(20):
            origin_sample = copy.deepcopy(sample)
            data_samples.append(sample)
        # dis = sample['label']
        # origin_sample = copy.deepcopy(sample)
        # data_samples.append(origin_sample)
        # imp_set = {}
        # mask = np.random.rand(len(sample['imp_sxs'].keys())) < random_masking
        # # print(mask)
        # i = 0
        # for k, v in sample['imp_sxs'].items():
        #     if mask[i] == True:
        #         # candidate = [sx for sx in dis2sx[dis] if sx not in sample['exp_sxs'].keys() and sx not in sample['imp_sxs'].keys()]
        #         # imp_set[random.choice(candidate)] = v
        #         imp_set[k] = v
        #     else:
        #         imp_set[k] = v
        #     i = i + 1
        # sample['imp_sxs'] = imp_set
        # data_samples.append(sample)
        # print(data_samples)
        # assert 0
    return data_samples

def data_augmentation_traindata(samples: list = None, real_prior_feat = None):
    # 增强后的训练数据集
    data_samples = []
    dis2sx = {}
    # 记录各疾病下的症状集
    for sample in samples:
        dis = sample['label']
        if dis not in dis2sx.keys():
            dis2sx[dis] = []
        for k, v in sample['imp_sxs'].items():
            if k not in dis2sx[dis]:
                if v == '1':
                    sy_sta = k + '-True'
                else:
                    sy_sta = k + '-False'
                if sy_sta in real_prior_feat:
                    dis2sx[dis].append(k)
    import copy
    for sample in samples:
        pass
        
        # dis = sample['label']
        # origin_sample = copy.deepcopy(sample)
        # data_samples.append(origin_sample)
        # imp_set = {}
        # mask = np.random.rand(len(sample['imp_sxs'].keys())) < random_masking
        # # print(mask)
        # i = 0
        # for k, v in sample['imp_sxs'].items():
        #     if mask[i] == True:
        #         # candidate = [sx for sx in dis2sx[dis] if sx not in sample['exp_sxs'].keys() and sx not in sample['imp_sxs'].keys()]
        #         # imp_set[random.choice(candidate)] = v
        #         imp_set[k] = v
        #     else:
        #         imp_set[k] = v
        #     i = i + 1
        # sample['imp_sxs'] = imp_set
        # data_samples.append(sample)
        # print(data_samples)
        # assert 0
    return data_samples
import numpy as np
import pickle
import json
import copy
import os
import random

def enrich_data(train_set):
    re_train_set = copy.deepcopy(train_set)
    for item in re_train_set:
        # 以0.6概率选取隐式症状，选1次
        for i in range(1):
            imp = {}
            for key,v in item['imp_sxs'].items():
                greedy = random.random()
                if greedy < 0.8:
                                 
                    imp[key] = v 
            item['imp_sxs'] = imp
            train_set.append(item)

    return train_set
# if self.train_mode is True and random.random() < 0.5:
        #     mask = np.random.rand(len(exp_sx_ids)) < random_masking

        #     input_sequence = np.array(exp_sx_ids)
        #     input_sequence[mask] = 0  # 使用0来表示遮挡
        #     exp_sx_ids= list(input_sequence)
        #     input_sequence = np.array(exp_attr_ids)
        #     input_sequence[mask] = 0  # 使用0来表示遮挡
        #     exp_attr_ids= list(input_sequence)
# 构造图
def create_graph(sx_ids, attr_ids, sv):
    edge_label_index = [[],[]]
    seq_len = sx_ids.size()[1]
    node_id = 0
    nodes = []
    for seq_id in range(sx_ids.size()[0]):
        valid_nodes = []
        make_edge = 0
        # if seq_id == 0:
        #     print(sx_ids[seq_id])
        #     print(attr_ids[seq_id])
        for indx in range(seq_len):
            if sx_ids[seq_id][indx].item() == 1:
                make_edge = 1
            if attr_ids[seq_id][indx].item() == 1 or attr_ids[seq_id][indx].item() == 2:
                act = sx_ids[seq_id][indx].item()
                attr = attr_ids[seq_id][indx].item()
                if (act in sv.prior_sx_attr.keys() and attr == sv.prior_sx_attr[act]) or \
                            (act in sv.prior_sx_attr_2.keys() and attr == sv.prior_sx_attr_2[act]):
                    for noded_id in valid_nodes:
                        edge_label_index[1].append(noded_id)
                        edge_label_index[0].append(node_id)
                        # edge_label_index[1].append(noded_id)
                        # edge_label_index[0].append(node_id)
                elif make_edge == 1:
                    for noded_id in valid_nodes:
                        edge_label_index[1].append(noded_id)
                        edge_label_index[0].append(node_id) 
                        # edge_label_index[1].append(noded_id)
                        # edge_label_index[0].append(node_id)    

                valid_nodes.append(node_id)
            nodes.append(sx_ids[seq_id][indx].item())
            
            node_id = node_id + 1
        # if seq_id == 0 :
        #     print(edge_label_index)
    return data.Data(edge_index=torch.LongTensor(edge_label_index), x=torch.LongTensor(nodes))

def create_train_graph(train_ds):
    edge_label_index = [[],[]]
    edge_attr = []
    nums= {}
    cur_nums = {}
    # 3 5
    for sample in train_ds:
        exp_sx_ids = sample['exp_sx_ids']
        imp_sx_ids = sample['imp_sx_ids'][1:-1]
        key_imp_sx_ids = sample['key_imp_sx_ids']
        valid_nodes = []
        for i in exp_sx_ids.cpu().tolist():
            # i->j
            for j in imp_sx_ids.cpu().tolist():
                ssstr = str(i)+'-'+str(j)
                if ssstr not in nums.keys():
                    nums[ssstr] = 1
                else:
                    nums[ssstr] = nums[ssstr]+1
                if nums[ssstr] == 3:
                    edge_label_index[0].append(i)
                    edge_label_index[1].append(j)
        for a in imp_sx_ids.cpu().tolist():
            # a->b,b->a
            for b in valid_nodes:
                ssstr = str(a)+'-'+str(b)
                if ssstr not in nums.keys():
                        nums[ssstr] = 1
                else:
                    nums[ssstr] = nums[ssstr]+1
                if nums[ssstr] == 3:
                    edge_label_index[0].append(a)
                    edge_label_index[1].append(b)
                    edge_label_index[1].append(a)
                    edge_label_index[0].append(b)
            valid_nodes.append(a)
    # for sample in train_ds:
    #     exp_sx_ids = sample['exp_sx_ids']
    #     imp_sx_ids = sample['imp_sx_ids'][1:-1]
    #     key_imp_sx_ids = sample['key_imp_sx_ids']
    #     valid_nodes = []
    #     for i in exp_sx_ids.cpu().tolist():
    #         # i->j
    #         for j in imp_sx_ids.cpu().tolist():
    #             ssstr = str(i)+'-'+str(j)
    #             if ssstr not in cur_nums.keys():
    #                 cur_nums[ssstr] = 1
    #             else:
    #                 cur_nums[ssstr] = cur_nums[ssstr]+1
    #             if nums[ssstr] == cur_nums[ssstr] and nums[ssstr] >= 10:
    #                 edge_label_index[0].append(i)
    #                 edge_label_index[1].append(j)
    #                 attr = 0
    #                 if nums[ssstr] <= 10:
    #                     attr = 0
    #                 elif nums[ssstr] <= 50:
    #                     attr = 1
    #                 elif nums[ssstr] <= 250:
    #                     attr = 2
    #                 elif nums[ssstr] <= 500:
    #                     attr = 3
    #                 else:
    #                     attr = 4
    #                 edge_attr.append(attr)
    #     for a in imp_sx_ids.cpu().tolist():
    #         # a->b,b->a
    #         for b in valid_nodes:
    #             ssstr = str(a)+'-'+str(b)
    #             if ssstr not in cur_nums.keys():
    #                     cur_nums[ssstr] = 1
    #             else:
    #                 cur_nums[ssstr] = cur_nums[ssstr]+1
    #             if nums[ssstr] == cur_nums[ssstr] and nums[ssstr] >= 10:
    #                 edge_label_index[0].append(a)
    #                 edge_label_index[1].append(b)
    #                 edge_label_index[1].append(a)
    #                 edge_label_index[0].append(b)
    #                 attr = 0
    #                 if nums[ssstr] <= 10:
    #                     attr = 0
    #                 elif nums[ssstr] <= 50:
    #                     attr = 1
    #                 elif nums[ssstr] <= 250:
    #                     attr = 2
    #                 elif nums[ssstr] <= 500:
    #                     attr = 3
    #                 else:
    #                     attr = 4
    #                 edge_attr.append(attr)
    #                 edge_attr.append(attr)
                    
    #         valid_nodes.append(a)
    
    # edge_nums = len(edge_attr)
    # edge_attr=torch.LongTensor(edge_attr).view(-1,1)
    return data.Data(edge_index=torch.LongTensor(edge_label_index),x=torch.LongTensor(list(range(0,gnn_nodes_num))))

def clean_dxy(train_samples, test_samples, real_prior_feat):
    for sample in train_samples:
        exp_set = {}
        for k, v in sample['exp_sxs'].items():
            if v == '1':
                sy_sta = k + '-True'
            else:
                sy_sta = k + '-False'
            if sy_sta in real_prior_feat:
                exp_set[k] = v
        sample['exp_sxs'] = exp_set
        imp_set = {}
        for k, v in sample['imp_sxs'].items():
            if v == '1':
                sy_sta = k + '-True'
            else:
                sy_sta = k + '-False'
                v = '0'
            if sy_sta in real_prior_feat:
                imp_set[k] = v
        sample['imp_sxs'] = imp_set
    for sample in test_samples:
        exp_set = {}
        for k, v in sample['exp_sxs'].items():
            if v == '1':
                sy_sta = k + '-True'
            else:
                sy_sta = k + '-False'
            if sy_sta in real_prior_feat:
                exp_set[k] = v
        sample['exp_sxs'] = exp_set
        imp_set = {}
        for k, v in sample['imp_sxs'].items():
            if v == '1':
                sy_sta = k + '-True'
            else:
                sy_sta = k + '-False'
                v = '0'
            if sy_sta in real_prior_feat:
                imp_set[k] = v
        sample['imp_sxs'] = imp_set
    return train_samples, test_samples
