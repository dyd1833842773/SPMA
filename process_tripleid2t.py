
train2id_file = r'FB15k237-owe-data\train2id.txt'
test2id_file = r'FB15k237-owe-data\test2id.txt'
valid2id_file = r'FB15k237-owe-data\valid2id.txt'
en2id_file = r'FB15k237-owe-data\entity2id.txt'
rel2id_file = r'FB15k237-owe-data\relation2id.txt'

train_file = r'FB15k237-owe-data\train.txt'
test_file = r'FB15k237-owe-data\test.txt'
valid_file = r'FB15k237-owe-data\valid.txt'

id2en = {}
id2rel = {}

with open(en2id_file,'r') as f:
    for l in f:
        e,id = l.strip().split('\t')
        id2en[id] = e

with open(rel2id_file,'r') as f:
    for l in f:
        r,id = l.strip().split('\t')
        id2rel[id] = r


with open(train_file,'w') as f2,open(train2id_file,'r') as f3:
    for l in f3:
        h,t,r = l.strip().split('\t')
        h_,r_,t_ = id2en[h],id2rel[r],id2en[t]
        f2.write(f'{h_}\t{r_}\t{t_}\n')

with open(test_file,'w') as f2,open(test2id_file,'r') as f3:
    for l in f3:
        h,t,r = l.strip().split('\t')
        h_,r_,t_ = id2en[h],id2rel[r],id2en[t]
        f2.write(f'{h_}\t{r_}\t{t_}\n')

with open(valid_file,'w') as f2,open(valid2id_file,'r') as f3:
    for l in f3:
        h,t,r = l.strip().split('\t')
        h_,r_,t_ = id2en[h],id2rel[r],id2en[t]
        f2.write(f'{h_}\t{r_}\t{t_}\n')