import os
import numpy as np
from PIL import Image
import cPickle as pickle
import pdb

import torch
from torchvision import transforms

def expand_tag_to_1006(tag_index_81_in_1006, label):
    tag_index_81_in_1006 = np.asarray(tag_index_81_in_1006)

    tag1006 = np.zeros(1006)
    tag1006[tag_index_81_in_1006] = label[:]

    pos_weights = np.zeros(1006)
    neg_weights = np.zeros(1006)
    weights = np.zeros(1006)

    pos_index = tag_index_81_in_1006[np.where(label == 1)[0]]
    neg_index = tag_index_81_in_1006[np.where(label == 0)[0]]

    weights[tag_index_81_in_1006] = 1.0

    pos_weights[pos_index] = 1.0
    neg_weights[neg_index] = 1.0

    return tag1006, pos_weights, neg_weights, weights

def accessing_row_elements_in_sparse(coo_sp, row_index):
    #pdb.set_trace()
    row, col = coo_sp.shape
    row_elements = np.zeros((col))
    specific_row_inds = np.where((coo_sp.row == row_index))[0]
    for col_,d in zip(coo_sp.col[specific_row_inds], coo_sp.data[specific_row_inds]):
        row_elements[col_] = d

    '''
    # test
    sp = coo_sp.toarray()
    array_row_elements = np.zeros((1, col))
    array_row_elements[0, :] = sp[row_index, :]
    assert ((row_elements == array_row_elements).all())
    '''
    return row_elements

def obtain_per_cls_weight(pos_cls_weights, neg_cls_weights, label):
    # pos_cls_weight: exp((1-p_c)/0.2)
    # neg_cls_weight: exp(pc/0.2)
    # input: label (1006,)
    # output: weights (1006,)
    '''
    num_classes = label.shape[0]
    weights = np.zeros(num_classes)
    weights[:] = neg_cls_weights[:]
    pos_ind = np.where(label == 1)[0]
    weights[pos_ind] = pos_cls_weights[pos_ind]
    '''
    #'''
    num_classes = label.shape[0]
    pos_weights = np.zeros(num_classes)
    neg_weights = np.zeros(num_classes)
    pos_ind = np.where(label == 1)[0]
    neg_ind = np.where(label == 0)[0]
    pos_weights[pos_ind] = 1
    neg_weights[neg_ind] = 1
    #'''
    return pos_weights, neg_weights

class extrDataset(torch.utils.data.Dataset):
    def __init__(self, root, scale_size):

        self.root = root
        self.paths = [os.path.join(self.root, line.strip().split()[0]) for line in open('/home/liyan/DeepLearning/ML-ZSL/ImageList/NUS-WIDE_Images_200k.txt', 'r')]

        taglist_1006 = [line.strip().split()[0] for line in open('./data_preprocessing/TagList1006.txt', 'rb')]
        taglist_81 = [line.strip().split()[0] for line in open('./data_preprocessing/Concepts81.txt', 'rb')]

        self.tag_index_81_in_1006 = [taglist_1006.index(tag_) for tag_ in taglist_81]

        pdb.set_trace()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.Resize([scale_size, scale_size]),
            transforms.ToTensor(),
            normalize,
            ])
    
    def __getitem__(self, index):
        #pdb.set_trace()
        image = Image.open(self.paths[index]).convert('RGB')
        return self.transform(image), self.paths[index]

    def __len__(self):
        return len(self.paths)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, scale_size, TEST=False, FEA=False):

        self.root = root
        self.fea = FEA

        #with open('./dataset/pos_81cls_weights.pkl', 'rb') as f:
        #    self.pos_cls_weights = pickle.load(f)
        #with open('./dataset/neg_81cls_weights.pkl', 'rb') as f:
        #    self.neg_cls_weights = pickle.load(f)
        
        taglist_1006 = [line.strip().split()[0] for line in open('./data_preprocessing/TagList1006.txt', 'rb')]
        taglist_81 = [line.strip().split()[0] for line in open('./data_preprocessing/Concepts81.txt', 'rb')]

        self.tag_index_81_in_1006 = [taglist_1006.index(tag_) for tag_ in taglist_81]
        self.tag_index_925_in_1006 = list(set(range(1006)) - set(self.tag_index_81_in_1006))
        weights = np.zeros(1006)
        weights[self.tag_index_925_in_1006] = 1.0
        self.weights = weights

        if TEST:
            self.paths = [os.path.join(self.root, line.strip().split()[0]) for line in open('./ImageList/NUS-WIDE_Images_test.txt', 'r')]

            print("Test Images: {0}".format(len(self.paths)))
            with open('./dataset/SparseNUSTestTags1006.pkl', 'rb') as f:
                sp_nustest_tags = pickle.load(f)
            self.sp_labels = sp_nustest_tags
        
        else:
            self.paths = [os.path.join(self.root, line.strip().split()[0]) for line in open('./ImageList/NUS-WIDE_Images_train.txt', 'r')]

            print("Train Images: {0}".format(len(self.paths)))
            with open('./dataset/SparseNUSTrainTags1006.pkl', 'rb') as f:
                sp_nustrain_tags = pickle.load(f)
            self.sp_labels = sp_nustrain_tags

            '''
            pdb.set_trace()
            # test
            nus_tags = self.sp_labels.toarray()
            for i in range(nus_tags.shape[0]):
                assert(nus_tags[i].sum()>0)
                assert((nus_tags[i]>=0).all())

            pdb.set_trace()
            #'''
        self.fea_paths = ['.'.join((line.strip().split('.')[0], line.strip().split('.')[1], 'fea')) for line in self.paths]
        #pdb.set_trace()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(scale_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        
        '''
        self.transform = transforms.Compose([
            transforms.Resize([scale_size, scale_size]),
            transforms.ToTensor(),
            normalize,
            ])
        '''

    def __getitem__(self, index):
        if self.fea:
            #print index
            #print self.fea_paths[index]
            #pdb.set_trace()
            with open(self.fea_paths[index], 'rb') as f:
                img_emb = pickle.load(f)
            #label = accessing_row_elements_in_sparse(self.sp_labels, index) #81
            #tag, _, _, weights = expand_tag_to_1006(self.tag_index_81_in_1006, label) #81 to 1006
            tag = accessing_row_elements_in_sparse(self.sp_labels, index) #1006
            weights = self.weights
            return torch.FloatTensor(img_emb), torch.FloatTensor(tag), torch.FloatTensor(weights)
        
        else:
            image = Image.open(self.paths[index]).convert('RGB')
            # accessing th index-th row labels from sparse attribuutes
            label = accessing_row_elements_in_sparse(self.sp_labels, index)
            tag, _, _, weights = expand_tag_to_1006(self.tag_index_81_in_1006, label)
            return self.transform(image), torch.FloatTensor(label), torch.FloatTensor(weights)
    
    '''
    def __getitem__(self, index):
        #pdb.set_trace()
        image = Image.open(self.paths[index]).convert('RGB')
        # accessing th index-th row labels from sparse attribuutes
        label = accessing_row_elements_in_sparse(self.sp_labels, index)
        return self.transform(image), self.paths[index]
    '''

    def __len__(self):
        return len(self.paths)


def get_loader(root, batch_size, scale_size, num_workers, ifshuffle, TEST, FEA):
    #pdb.set_trace()
    data_set = Dataset(root, scale_size, TEST, FEA)

    data_loader = torch.utils.data.DataLoader(dataset=data_set,
                                              batch_size=batch_size,
                                              shuffle=ifshuffle,
                                              num_workers=num_workers)

    return data_loader


# Unit Tests
if __name__ == '__main__':
    pdb.set_trace()
    with open('./dataset/SparseNUS90kTags1006.pkl', 'rb') as f:
        coo_sp = pickle.load(f)

    single_label = accessing_row_elements_in_sparse(coo_sp, 10)
