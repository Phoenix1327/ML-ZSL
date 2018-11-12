import os
import numpy as np
from nltk.corpus import wordnet as wn
import pdb

TagsAll = [line.strip().split()[0] for line in open('TagList1006.txt')]

pdb.set_trace()

num_tags = len(TagsAll)

file = open('tagname_wordnet.txt', 'w')

for i in range(num_tags):
    tag_name = TagsAll[i]
    syn_list = wn.synsets(tag_name)
    num_syns = len(syn_list)

    for j in range(num_syns):
        syn_def = syn_list[j].definition()
        syn_name = syn_list[j].name()
        print "synset:----{0}, name: {1}, \ndefinition: {2}".format(j, syn_name, syn_def)

    decision = raw_input()
    print "######## \n Make the decision: {0} \n ########".format(decision)

    selected_synset_name = syn_list[int(decision)].name()
    
    write_data = tag_name + ' ' + selected_synset_name + '\n'
    file.write(write_data)

file.close()

