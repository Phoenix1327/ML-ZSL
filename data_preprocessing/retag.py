import os
import numpy as np
import scipy.sparse as sp
import cPickle
import pdb

TagsAll = [line.strip().split()[0] for line in open('TagList1006.txt')]
Tags1k = [line.strip().split()[0] for line in open('TagList1k.txt')]
Tags81 = [line.strip().split()[0] for line in open('Concepts81.txt')]

pdb.set_trace()
num_tags = len(TagsAll)

labels_1000 = np.loadtxt('AllTags1k.txt').astype(int)
labels_81 = np.loadtxt('AllTags81.txt').astype(int)
num_images = labels_81.shape[0]

print "Load 1000-tag and 81-tag labels "

AllTags1006 = np.zeros((num_images, num_tags), dtype=np.int)
for i in range(num_images):
    print i

    single_label_1000 = labels_1000[i]
    single_label_81 = labels_81[i]

    # 1000 labels indices
    #'''
    #pdb.set_trace()
    indices_1000 = [k for k, x in enumerate(single_label_1000) if x == 1]
    #pdb.set_trace()
    for ind in indices_1000:
        tag_name = Tags1k[ind]
        # find index in 1006 tags
        ind_1006 = TagsAll.index(tag_name)
        AllTags1006[i, ind_1006] = 1
    #'''
    # 81 labels indices
    #pdb.set_trace()
    indices_81 = [k for k, x in enumerate(single_label_81) if x == 1]
    #pdb.set_trace()
    for ind in indices_81:
        tag_name = Tags81[ind]
        # find index in 1006 tags
        ind_1006 = TagsAll.index(tag_name)
        AllTags1006[i, ind_1006] = 1

    # store the image of each image
    #single_tag_1006[0,:] = AllTags1006[i,:]
    #label_file = "AllTags1006/{}.txt".format(i+1)
    #np.savetxt(label_file, single_tag_1006, fmt='%d')

# convert the label matrix to sparse matrix
sp_AllTags1006 = sp.coo_matrix(AllTags1006)
with open('./SparseAllTags1006.pkl', 'wb') as f:
    cPickle.dump(sp_AllTags1006, f, cPickle.HIGHEST_PROTOCOL)
