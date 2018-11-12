import os
import numpy as np
import scipy.sparse as sp
import copy
import random
import cPickle
import pdb

#alltag = np.loadtxt('./AllTags81.txt').astype(int)
all_image = ['/'.join(line.strip().split('\\')) for line in open('../ImageList/Imagelist.txt', 'r')]

pdb.set_trace()

num_images = 269648
num_classes = 81
gttag = np.zeros((num_images, num_classes))


for line in open('./Concepts81.txt', 'r'):
    print line

tag_names = [line.strip().split()[0] for line in open('./Concepts81.txt', 'r')]

for i in range(num_classes):
    tag_ = tag_names[i]
    taggt_file = '/home/liyan/DeepLearning/ML-ZSL/ImageList/AllLabels/Labels_' + tag_ + '.txt'
    #pdb.set_trace()
    label_ = [int(line.strip().split()[0]) for line in open(taggt_file, 'r')]
    label_ = np.asarray(label_)
    gttag[:, i] = label_[:]


#'''
#filter_img = []
filtergt_img = []
for i in range(len(all_image)):
    #tag_ = alltag[i]
    taggt_ = gttag[i]
    #if np.sum(tag_) > 0:
    #    filter_img.append(all_image[i])
    #    assert((tag_ == taggt_).all())
    if np.sum(taggt_) > 0:
        filtergt_img.append(all_image[i])
#'''

print len(filtergt_img)

filtergt_index = [all_image.index(img_) for img_ in filtergt_img]
filter_gttag = gttag[filtergt_index, :] #209347*81
sp_filter_gttag = sp.coo_matrix(filter_gttag)

with open('/home/liyan/DeepLearning/ML-ZSL/dataset/SparseNUS200kAllTags81.pkl', 'wb') as f:
    cPickle.dump(sp_filter_gttag, f, cPickle.HIGHEST_PROTOCOL)

f = open('/home/liyan/DeepLearning/ML-ZSL/ImageList/NUS-WIDE_Images_200k.txt', 'w')
for img_ in filtergt_img:
    f.write(img_ + '\n')
f.close()


pdb.set_trace()

Ori_filtergt_img = copy.deepcopy(filtergt_img) #209347, orginal
random.shuffle(filtergt_img)

Train_list = filtergt_img[:150000]
Val_list = filtergt_img[:10000]
Test_list = filtergt_img[150000:]


f = open('/home/liyan/DeepLearning/ML-ZSL/ImageList/NUS-WIDE_Images_200k_train.txt', 'w')
for img_ in Train_list:
    f.write(img_ + '\n')
f.close()


pdb.set_trace()

f = open('/home/liyan/DeepLearning/ML-ZSL/ImageList/NUS-WIDE_Images_200k_val.txt', 'w')
for img_ in Val_list:
    f.write(img_ + '\n')
f.close()

pdb.set_trace()

f = open('/home/liyan/DeepLearning/ML-ZSL/ImageList/NUS-WIDE_Images_200k_test.txt', 'w')
for img_ in Test_list:
    f.write(img_ + '\n')
f.close()

pdb.set_trace()

Train_index = [Ori_filtergt_img.index(img_) for img_ in Train_list]
Val_index = [Ori_filtergt_img.index(img_) for img_ in Val_list]
Test_index = [Ori_filtergt_img.index(img_) for img_ in Test_list]

pdb.set_trace()

traintag81 = filter_gttag[Train_index, :]
sp_traintag81 = sp.coo_matrix(traintag81)

valtag81 = filter_gttag[Val_index, :]
sp_valtag81 = sp.coo_matrix(valtag81)

testtag81 = filter_gttag[Test_index, :]
sp_testtag81 = sp.coo_matrix(testtag81)


with open('/home/liyan/DeepLearning/ML-ZSL/dataset/SparseNUS200kTrainTags81.pkl', 'wb') as f:
    cPickle.dump(sp_traintag81, f, cPickle.HIGHEST_PROTOCOL)

with open('/home/liyan/DeepLearning/ML-ZSL/dataset/SparseNUS200kValTags81.pkl', 'wb') as f:
    cPickle.dump(sp_valtag81, f, cPickle.HIGHEST_PROTOCOL)

with open('/home/liyan/DeepLearning/ML-ZSL/dataset/SparseNUS200kTestTags81.pkl', 'wb') as f:
    cPickle.dump(sp_testtag81, f, cPickle.HIGHEST_PROTOCOL)


pdb.set_trace()

