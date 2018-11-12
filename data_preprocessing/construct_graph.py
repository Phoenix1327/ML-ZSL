import os
import numpy as np
from nltk.corpus import wordnet as wn
import cPickle as pickle
import pdb

TagsAll = [line.strip().split()[0] for line in open('tagname_wordnet.txt')]
SynsAll = [line.strip().split()[1] for line in open('tagname_wordnet.txt')]
pdb.set_trace()

num_tags = len(TagsAll)
Graph = np.zeros((num_tags, num_tags))
wup_sim_thresh = 0.75


for i in range(num_tags):
    print i
    for j in range(i,num_tags):
        i_wn = SynsAll[i]
        i_wn = wn.synset(i_wn)
        j_wn = SynsAll[j]
        j_wn = wn.synset(j_wn)

        super_sub_flag = False
        hypoi_wn = set([m for m in i_wn.closure(lambda s:s.hyponyms())])
        hypoj_wn = set([n for n in j_wn.closure(lambda s:s.hyponyms())])
        if j_wn in hypoi_wn:
            super_sub_flag = True
        if i_wn in hypoj_wn:
            super_sub_flag = True

        if super_sub_flag:
            #pdb.set_trace()
            #print "word i---definition: {}".format(i_wn.definition())
            #print "word j---definition: {}".format(j_wn.definition())
            Graph[i,j] = 1
            Graph[j,i] = 1

        wup_sim = wn.wup_similarity(i_wn, j_wn)
        if wup_sim > wup_sim_thresh:
            #if wup_sim < 1.0:
                #print "word i---definition: {}".format(i_wn.definition())
                #print "word j---definition: {}".format(j_wn.definition())
            Graph[i,j] = 1
            Graph[j,i] = 1

#pdb.set_trace()

print "The Graph is constructed, the number of edges is: {}".format(Graph.sum()-num_tags)

graph_file = "AllTags1006Adj.pkl"
with open(graph_file, 'wb') as f:
    pickle.dump(Graph, f, pickle.HIGHEST_PROTOCOL)
