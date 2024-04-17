import pickle

import json
with open('goal_set.p', 'rb') as f:
    data = pickle.load(f)
train_json = []
test_json = []

for goal in data['train']:
    sample = {}
    exp_sxs = {}
    imp_sxs = {}
    sample['pid'] = goal['consult_id']
    sample['label'] = goal['disease_tag']
    for k,v in goal['goal']['explicit_inform_slots'].items():
        if v == True:
           exp_sxs[k] = '1' 
        elif v == False:
            exp_sxs[k] = '0'
    for k,v in goal['goal']['implicit_inform_slots'].items():
        if v == True:
           imp_sxs[k] = '1' 
        elif v == False:
            imp_sxs[k] = '0'
    sample['imp_sxs'] = imp_sxs
    sample['exp_sxs'] = exp_sxs
    train_json.append(sample)
    
for goal in data['test']:
    sample = {}
    exp_sxs = {}
    imp_sxs = {}
    sample['pid'] = goal['consult_id']
    sample['label'] = goal['disease_tag']
    for k,v in goal['goal']['explicit_inform_slots'].items():
        if v == True:
           exp_sxs[k] = '1' 
        elif v == False:
            exp_sxs[k] = '0'
    for k,v in goal['goal']['implicit_inform_slots'].items():
        if v == True:
           imp_sxs[k] = '1' 
        elif v == False:
            imp_sxs[k] = '0'
    sample['imp_sxs'] = imp_sxs
    sample['exp_sxs'] = exp_sxs
    test_json.append(sample)
f.close()
with open('train_set.json', 'w') as f:
    json.dump(train_json, f, ensure_ascii=False)

with open('test_set.json', 'w') as f:
    json.dump(test_json, f, ensure_ascii=False)
