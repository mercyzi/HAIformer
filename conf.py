# This is all the parameters. 
import argparse

# parameters (training)
parser = argparse.ArgumentParser()
parser.add_argument('-data', '--train_dataset', default='MDD', help='choose the training dataset')
args = parser.parse_args()

# dataset for training
train_dataset = args.train_dataset

# dataset for testing
test_dataset = train_dataset

# check validity
ds_range = ['dxy','mz10', 'muzhi', 'MDD']

assert train_dataset in ds_range
assert test_dataset in ds_range

device_num = 0

# train/test data path
train_path = []
if train_dataset != 'all':
    train_path.append('data/{}/train_set.json'.format(train_dataset))
else:
    for ds in ds_range[1:]:
        train_path.append('data/{}/train_set.json'.format(ds))

test_path = []
if test_dataset != 'all':
    test_path.append('data/{}/test_set.json'.format(test_dataset))
else:
    for ds in ds_range[1:]:
        test_path.append('data/{}/test_set.json'.format(ds))

best_pt_path = 'saved/{}/best_pt_exe_model.pt'.format(train_dataset)
last_pt_path = 'saved/{}/last_pt_exe_model.pt'.format(train_dataset)

pre_gnn_path = 'data/{}/pregnn.pt'.format(train_dataset)
gnn_nodes_num = {'dxy': 47, 'mz10': 322, 'muzhi': 72, 'MDD': 122}.get(train_dataset)

# global settings
suffix = {'0': '-Negative', '1': '-Positive', '2': '-Negative'}
min_sx_freq = None
max_voc_size = None
keep_unk = True
digits = 4

# model hyperparameter setting
pos_dropout = 0.1
pos_max_len = 80

sx_one_hot = False
attr_one_hot = False
num_attrs = 5
num_executor = 2

dec_emb_dim = 256 if train_dataset == 'dxy' else 512
dec_dim_feedforward = 2 * dec_emb_dim

dec_num_heads = 8
dec_num_layers = 4
dec_dropout = 0.5


exclude_exp = False
only_key = False
# group 3: transformer encoder
enc_emb_dim = dec_emb_dim
enc_dim_feedforward = 2 * enc_emb_dim

enc_num_heads = 8 if train_dataset == 'dxy' or train_dataset == 'MDD' else 4
enc_num_layers = 2 if train_dataset == 'dxy' or train_dataset == 'MDD' else 1
enc_dropout = {'dxy': 0.75, 'mz10': 0.6, 'muzhi': 0.25, 'MDD': 0.75}.get(train_dataset)

# group 3: training
num_workers = 0

pt_ratio = {'dxy': 1, 'mz10': 1, 'muzhi': 1, 'MDD': 1}.get(train_dataset)
pt_train_epochs = 200
pt_learning_rate = {'dxy': 3e-4, 'mz10': 1e-4, 'muzhi': 9e-5, 'MDD': 3e-4}.get(train_dataset)
pt_learning_save = {'dxy': 0.8, 'mz10': 0.65, 'muzhi': 0.73, 'MDD': 0.85}.get(train_dataset)

learning_rate = 1e-4 
train_epochs = {'dxy': 100, 'mz10': 100, 'muzhi': 100, 'MDD': 100}.get(train_dataset)
warm_epoch = train_epochs #// 2

train_bsz = 128
test_bsz = 128

alpha = 0.2
exp_name = 'hai_all'
num_turns = {'dxy': 20, 'mz10': 15, 'muzhi': 15, 'MDD': 10}.get(train_dataset)
num_repeats = 5
verbose = True


gamma = 1

capacity = 10


human_invlove = True
human_ability = 1
human_step_s = {'dxy': 2, 'mz10': 3, 'muzhi': 2, 'MDD': 2}.get(train_dataset)
human_step = {'dxy': 10, 'mz10': 15, 'muzhi': 10, 'MDD': 10}.get(train_dataset)
key_hp = 0.75


random_masking = 0.4
data_aug = False

emphasis_factor = 1.0
dec_add_pos = False

ratio1 = 1
ratio2 = 1
ratio3 = 1
ratio4 = 1

alpha_graph = 0.5
alpha_add = (1 - alpha_graph)/2

a_num = 0
# our AI/Doctor: disease_conf = {'dxy': 0.92, 'mz10': 0.996, 'muzhi': 0.995, 'MDD': 0.995}.get(train_dataset) 98 97 96 95 94
disease_conf = {'dxy': 0.98, 'mz10': 0.996, 'muzhi': 0.92, 'MDD': 0.86}.get(train_dataset)
symptom_conf = {'dxy': 0.97, 'mz10': 0.97, 'muzhi': 0.97, 'MDD': 0.97}.get(train_dataset)
train_dc_epoch = {'dxy': 30, 'mz10': 30, 'muzhi': 45, 'MDD': 20}.get(train_dataset)

prior_feat_nums = {'dxy': 16, 'mz10': 138, 'muzhi': 36, 'MDD': 116}.get(train_dataset)
prior_feat_path = 'data/{}_prior_feat.json'.format(train_dataset)
