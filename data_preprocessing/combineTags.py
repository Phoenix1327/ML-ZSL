import os
import numpy as np
import pdb

Tags1k = [line.strip().split()[0] for line in open('TagList1k.txt')]
Tags81 = [line.strip().split()[0] for line in open('Concepts81.txt')]

pdb.set_trace()

TagsAll = Tags81 + list(set(Tags1k) - set(Tags81))
TagsAll.sort()
print "Combine the tags of NUS-1000 and NUS-81, with removing duplicates"
print "Total tags: {}".format(len(TagsAll))

TagsSeen = list(set(Tags1k) - set(Tags81))
TagsSeen.sort()
print "Tags of Seen classes (NUS-1000 - NUS-81)"
print "Total tags: {}".format(len(TagsSeen))

TagsNew = list(set(TagsAll) - set(Tags1k))
print "Newly added tags"
print TagsNew

# write tags
TagsAllfile = open('TagList1006.txt', 'w')
for item in TagsAll:
    TagsAllfile.write("%s\n" % item)
TagsAllfile.close()

TagsSeenfile = open('TagList925.txt', 'w')
for item in TagsSeen:
    TagsSeenfile.write("%s\n" % item)
TagsSeenfile.close()
