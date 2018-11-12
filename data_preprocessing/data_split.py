import os
import copy
import numpy as np
import scipy.sparse as sp
import random
import cPickle as pickle

import pdb


root = '../NUS-WIDE/'
fulltrain_list = [os.path.join(root, line.strip().split()[0]) for line in open('../ImageList/NUS-WIDE_Images_train.txt', 'r')]

pdb.set_trace()

for file_ in fulltrain_list:
    assert(os.path.exists(file_))


pdb.set_trace()

train_list = [line.strip().split()[0] for line in open('../ImageList/NUS-WIDE_Images_train.txt', 'r')]

pdb.set_trace()
val_list = train_list[:5000]

f = open('../ImageList/NUS-WIDE_Images_val.txt', 'w')
for file_ in val_list:
    write_data = file_ + '\n'
    f.write(write_data)
f.close()

pdb.set_trace()

with open('./SparseNUSTrainTags1006.pkl', 'rb') as f:
    sp_traintags1006 = pickle.load(f)
    traintags1006 = sp_traintags1006.toarray()

valtags1006 = traintags1006[:5000]
sp_valtags1006 = sp.coo_matrix(valtags1006)

with open('./SparseNUSValTags1006.pkl', 'wb') as f:
    pickle.dump(sp_valtags1006, f, pickle.HIGHEST_PROTOCOL)

'''
Allpaths_list = [line.strip().split()[0] for line in open('../ImageList/NUS-WIDE_Images_90k.txt')]
Ori_Allpaths = copy.deepcopy(Allpaths_list)

#pdb.set_trace()

random.shuffle(Allpaths_list)

Trainpaths_list = Allpaths_list[:80000]
Testpaths_list = Allpaths_list[80000:]

Train_index = [Ori_Allpaths.index(img_) for img_ in Trainpaths_list]
Test_index = [Ori_Allpaths.index(img_) for img_ in Testpaths_list]

#pdb.set_trace()

with open('./SparseNUS90kTags1006.pkl', 'rb') as f:
    sp_alltags1006 = pickle.load(f)
    alltags1006 = sp_alltags1006.toarray() #90k*1006

Trainalltags = alltags1006[Train_index]
Testalltags = alltags1006[Test_index]

sp_trainalltags = sp.coo_matrix(Trainalltags)
sp_testalltags = sp.coo_matrix(Testalltags)

with open('./SparseNUSTrainTags1006.pkl', 'wb') as f:
    pickle.dump(sp_trainalltags, f, pickle.HIGHEST_PROTOCOL)

with open('./SparseNUSTestTags1006.pkl', 'wb') as f:
    pickle.dump(sp_testalltags, f, pickle.HIGHEST_PROTOCOL)


f = open('../ImageList/NUS-WIDE_Images_train.txt', 'w')
for img_ in Trainpaths_list:
    write_data = img_ + '\n'
    f.write(write_data)
f.close()

f = open('../ImageList/NUS-WIDE_Images_test.txt', 'w')
for img_ in Testpaths_list:
    write_data = img_ + '\n'
    f.write(write_data)
f.close()
'''
