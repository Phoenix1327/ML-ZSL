import os
import numpy as np
import cPickle as pickle
import pdb

with open('glove.840B.300d.txt', 'r') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = map(float, vals[1:])


#pdb.set_trace()
print "Load wordvectors successfully!"
TagsAll = [line.strip().split()[0] for line in open('TagList1006.txt')]
pdb.set_trace()
num_tags = len(TagsAll)
dim_vec = 300
tags_word2vector = np.zeros((num_tags, dim_vec)) # 1006*300
tags_w2v_file = "AllTags1006W2V.pkl"

for i in range(num_tags):
    tag_ = TagsAll[i]
    w2v = vectors[tag_]
    w2v = np.asarray(w2v)
    tags_word2vector[i,:] = w2v[:]

with open(tags_w2v_file, 'wb') as f:
    pickle.dump(tags_word2vector, f, pickle.HIGHEST_PROTOCOL)

print "Extract wordvectors for all tags"
