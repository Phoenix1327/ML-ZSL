import os
import numpy as np
import scipy.sparse as sp
import cPickle
import pdb

taglist_1006 = [line.strip().split()[0] for line in open('./TagList1006.txt', 'rb')]
taglist_81 = [line.strip().split()[0] for line in open('./Concepts81.txt', 'rb')]

tag_index_81_in_1006 = [taglist_1006.index(tag_) for tag_ in taglist_81]

'''
with open('../dataset/SparseNUSTrainTags1006.pkl', 'rb') as f:
    sp_traintags1006 = cPickle.load(f)
traintags1006 = sp_traintags1006.toarray()
traintags81 = np.zeros((traintags1006.shape[0], 81))
traintags81[:, :] = traintags1006[:, tag_index_81_in_1006]
sp_traintags81 = sp.coo_matrix(traintags81)
with open('../dataset/SparseNUSTrainTags81_fromAllTags81.pkl', 'wb') as f:
    cPickle.dump(sp_traintags81, f, cPickle.HIGHEST_PROTOCOL)

with open('../dataset/SparseNUSTestTags1006.pkl', 'rb') as f:
    sp_testtags1006 = cPickle.load(f)
testtags1006 = sp_testtags1006.toarray()
testtags81 = np.zeros((testtags1006.shape[0], 81))
testtags81[:, :] = testtags1006[:, tag_index_81_in_1006]
sp_testtags81 = sp.coo_matrix(testtags81)
with open('../dataset/SparseNUSTestTags81_fromAllTags81.pkl', 'wb') as f:
    cPickle.dump(sp_testtags81, f, cPickle.HIGHEST_PROTOCOL)
'''

with open('../dataset/SparseNUSValTags1006.pkl', 'rb') as f:
    sp_valtags1006 = cPickle.load(f)
valtags1006 = sp_valtags1006.toarray()
valtags81 = np.zeros((valtags1006.shape[0], 81))
valtags81[:, :] = valtags1006[:, tag_index_81_in_1006]
sp_valtags81 = sp.coo_matrix(valtags81)
with open('../dataset/SparseNUSValTags81_fromAllTags81.pkl', 'wb') as f:
    cPickle.dump(sp_valtags81, f, cPickle.HIGHEST_PROTOCOL)
