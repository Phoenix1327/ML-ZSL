import os
import numpy as np
import scipy.sparse as sp
import cPickle as pickle
import pdb

AllImagelist = ['/'.join(line.strip().split('\\')) for line in open('../ImageList/Imagelist.txt', 'r')]

NUS90kImageList = [line.strip().split()[0] for line in open('../ImageList/NUS-WIDE_Images_90k.txt', 'r')]

pdb.set_trace()
#find NUS90k indices in AllImage
inds = [AllImagelist.index(image) for image in NUS90kImageList]
pdb.set_trace()

with open('../dataset/SparseAllTags1006.pkl', 'rb') as f:
    #sparse matrix, coo
    #matrix.data storing non-zero values
    #matrix.row  corresponding row indexes
    #matrix.col  corresponding col indexes
    SparseAllTags1006 = pickle.load(f)

AllTags1006 = SparseAllTags1006.toarray()
NUS90kTags1006 = AllTags1006[inds, :]
sp_NUS90kTags1006 = sp.coo_matrix(NUS90kTags1006)

with open('./SparseNUS90kTags1006.pkl', 'wb') as f:
    pickle.dump(sp_NUS90kTags1006, f, pickle.HIGHEST_PROTOCOL)
