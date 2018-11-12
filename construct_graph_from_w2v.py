import os
import numpy as np
import cPickle
import pdb

with open('./dataset/AllTags1006W2V.pkl', 'rb') as f:
    w2v = cPickle.load(f) #1006*300
    w2v /= ((w2v**2).sum(axis=1, keepdims=True))**0.5

pdb.set_trace()

num_tags = w2v.shape[0]
Graph = np.zeros((num_tags, num_tags))

for i in range(num_tags):
    print i
    for j in range(i, num_tags):
        cos_sim = max(0, np.dot(w2v[i,:], w2v[j,:]))
        Graph[i, j] = cos_sim
        Graph[j, i] = cos_sim

pdb.set_trace()
with open('./dataset/AllTags1006Adj_fromw2v.pkl', 'wb') as f:
    cPickle.dump(Graph, f, cPickle.HIGHEST_PROTOCOL)

