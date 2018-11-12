import scipy.sparse as sp
import numpy as np
import cPickle as pickle

with open('./SparseAllTags1006.pkl', 'rb') as f:
    sp_AllTags1006 = pickle.load(f)
    AllTags1006 = sp_AllTags1006.toarray() #N*1006

AllImageListfile = './ImageList/Imagelist.txt'
FilteredImageListfile = './ImageList/FilteredImageList.txt'


fin = open(AllImageListfile, 'r')
allimages = fin.readlines()
fin.close()

fout = open(FilteredImageListfile, 'w')
num_images = len(allimages)

for i in range(num_images):
    if AllTags1006[i].sum() > 0:
        line = allimages[i]
        line = line.strip()
        fout.write(line+'\n')

fout.close()
