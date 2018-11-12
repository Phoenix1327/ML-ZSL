import os
import numpy as np
import scipy.sparse as sp
import cPickle as pickle

import pdb

with open('../dataset/SparseNUSTrainTags81.pkl', 'rb') as f:
    sp_nustrain_1006 = pickle.load(f)
    nustrain_1006 = sp_nustrain_1006.toarray()

num_classes = nustrain_1006.shape[1]
percentage = np.zeros((num_classes))

for i in range(num_classes):
    percentage[i] = float(np.sum(nustrain_1006[:,i])) / float(nustrain_1006.shape[0])


pdb.set_trace()
#assert (np.sum(percentage) == 1)

theta = 0.1
pos_cls_weights = 1 - percentage
pos_cls_weights = pos_cls_weights / theta
pos_cls_weights = np.exp(pos_cls_weights)

neg_cls_weights = percentage / theta
neg_cls_weights = np.exp(neg_cls_weights)

with open('../dataset/pos_81cls_weights.pkl', 'wb') as f:
    pickle.dump(pos_cls_weights, f, pickle.HIGHEST_PROTOCOL)

with open('../dataset/neg_81cls_weights.pkl', 'wb') as f:
    pickle.dump(neg_cls_weights, f, pickle.HIGHEST_PROTOCOL)


