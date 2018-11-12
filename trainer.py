from __future__ import print_function
from collections import defaultdict

import os
import numpy as np
import scipy.sparse as sp
import cPickle as pickle

import sklearn.metrics
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

import torch
from torch import nn
import torchvision
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.nn import functional as F

from models import GAE, VAE, CLS
from utils import load_data, dotdict, eval_gae, make_sparse, plot_results, get_subsampler, get_possampler
from preprocessing import mask_test_edges, preprocess_graph, prepare_graph

from data_loader import get_loader
import pdb


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.001)
        #m.bias.data.fill_(0)


class Trainer(object):
    def __init__(self, config, data_loader, test_data_loader):
        self.config = config

        self.data_loader = data_loader
        self.test_data_loader = test_data_loader

        self.num_gpu = config.num_gpu
        self.dataset = config.dataset

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.bath_size = config.batch_size
        self.weight_decay = config.weight_decay

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step

        # tag_index_81_in_1006
        taglist_1006 = [line.strip().split()[0] for line in open('./data_preprocessing/TagList1006.txt', 'rb')]
        taglist_81 = [line.strip().split()[0] for line in open('./data_preprocessing/Concepts81.txt', 'rb')]
        self.tag_index_81_in_1006 = [taglist_1006.index(tag_) for tag_ in taglist_81]
        self.tag_index_925_in_1006 = list(set(range(1006)) - set(self.tag_index_81_in_1006))
        #pdb.set_trace()
        # GAE data preprocessing
        adj_norm, fea, self.adj_labels, self.N, self.norm, weight = prepare_graph(self.tag_index_925_in_1006)
        # Some preprocessing
        #adj_norm = make_sparse(adj_norm)

        #pdb.set_trace()
        adj_norm = torch.FloatTensor(adj_norm.toarray())
        #adj_norm = self._get_variable(adj_norm)
        self.adj_norm = adj_norm

        fea = make_sparse(fea)
        #fea = torch.FloatTensor(fea)
        fea = self._get_variable(fea)
        self.fea = fea

        weight = torch.FloatTensor(weight)
        weight = self._get_variable(weight)
        self.weight = weight

        # Graph data preprocessing is done
        
        self.build_model()

        if self.num_gpu > 0:
            #self.fea_extractor.cuda()
            self.GAE.cuda()
            self.VAE.cuda()
            self.CLS.cuda()

        #self.CLS.apply(weights_init)
        
        optimizer = torch.optim.Adam
        self.optimizer_gae = optimizer(filter(lambda p: p.requires_grad, self.GAE.parameters()), lr=0.0001, betas=(0.95, 0.999))
        self.optimizer_vae = optimizer(filter(lambda p: p.requires_grad, self.VAE.parameters()), lr=0.0001, betas=(0.95, 0.999))
        self.optimizer_cls = optimizer(filter(lambda p: p.requires_grad, self.CLS.parameters()), lr=0.0005, betas=(0.95, 0.999), weight_decay=0.0001)
        #self.optimizer_cls = torch.optim.SGD(self.CLS.parameters(), lr=0.001, weight_decay=0.0001)

        if self.load_path:
            self.load_model()

    def build_model(self):

        '''
        self.fea_extractor = torchvision.models.resnet152(pretrained=True)
        for p in self.fea_extractor.parameters():
            p.requires_grad = False
        '''

        self.GAE = GAE(n_feat = 300,
                       n_hid = 256,
                       n_latent = 128,
                       adj = self.adj_norm,
                       dropout = 0.0)

        self.VAE = VAE()
        #pdb.set_trace()
        self.CLS = CLS(num_classes=1006)
        self.CLS.apply(weights_init)
        #self.CLS = torchvision.models.resnet152(pretrained=True)
        #self.CLS.fc = nn.Linear(in_features=2048, out_features=1006)


    def _extract_features(self, image_):
        # input: image_, Variable
        # output: img_emb, Variable

        batchsize = image_.size(0)
        # extract img features
        img_emb = torch.zeros((batchsize, 2048, 1, 1))
        def copy_data(m,i,o):
            img_emb.copy_(o.data)
        layer = self.fea_extractor._modules.get('avgpool')
        h = layer.register_forward_hook(copy_data)
        h_x = self.fea_extractor(image_)
        h.remove()
        img_emb = img_emb.view(batchsize, -1) #N*2048
        #print ("Features are extracted")
        return self._get_variable(img_emb)

    def _get_variable(self, inputs):
        if self.num_gpu > 0:
            out = Variable(inputs.cuda())
        else:
            out = Variable(inputs)
        return out

    def _gae_train(self, epoch, results):
        self.GAE.train()
        self.GAE.zero_grad()
        #pdb.set_trace()
        recon_adj, mu, std = self.GAE(self.fea, self.adj_norm)
        # Reconstruction + KL divergence losses summed over all elements
        gae_bce = F.binary_cross_entropy(recon_adj, self.adj_labels, self.weight, size_average=True) * self.norm
        gae_kld = -(0.5 / self.N) * torch.mean(torch.sum(1 + 2*torch.log(std) - mu.pow(2) - std.pow(2), 1))
        loss_gae = gae_bce + 0*gae_kld
        loss_gae.backward()
        self.optimizer_gae.step()

        results['gae_bce'].append(gae_bce.cpu().data[0])
        results['gae_kld'].append(gae_kld.cpu().data[0])
        results['loss_gae'].append(loss_gae.cpu().data[0])

        print("Optimization GAE: Epoch----{0}".format(epoch),
              "GAE loss: {:.5f}".format(loss_gae.cpu().data[0]),
              "BCE loss: {:.5f}".format(gae_bce.cpu().data[0]),
              "KLD loss: {:.5f}".format(gae_kld.cpu().data[0]))

        return results

    def _gae_eval(self):
        self.GAE.eval()
        _, graph_emb, _ = self.GAE(self.fea, self.adj_norm) #graph_emb: (1006*300)
        return graph_emb

    def _joint_train(self, epoch, results):
        print("Optimization Joint Training: Epoch----{0}".format(epoch))
        self.VAE.train()
        self.GAE.train()
        #pdb.set_trace()
        for i, (image_fea_, label_, weight_) in enumerate(self.data_loader):
            self.VAE.zero_grad()
            self.GAE.zero_grad()
            #pdb.set_trace()
            image_fea_ = self._get_variable(image_fea_) #N*2048
            label_ = self._get_variable(label_) #N*1006
            weight_ = self._get_variable(weight_) #N*1006
            _, graph_emb, _ = self.GAE(self.fea) #graph_emb: (1006*128)

            #pdb.set_trace()
            #L1 regularization for adj matrix
            #l1_regularization = torch.norm(self.GAE.gc1.adj, 1, 1).mean()
            #l1_regularization += torch.norm(self.GAE.gc2_mu.adj, 1, 1).mean()
            #l1_regularization *= 0.005


            #graph_emb = F.normalize(graph_emb, p=2, dim=1)
            _, image_emb, _ = self.VAE(image_fea_) #N*128
            cls_predict = torch.mm(image_emb, graph_emb.t()) #N*1006
            cls_predict = F.sigmoid(cls_predict)
            loss_img_bce = F.binary_cross_entropy(cls_predict, label_, weight_, size_average=True) / 925.0 * 1006

            #loss_img_bce += l1_regularization
            loss_img_bce.backward()
            self.optimizer_vae.step()
            self.optimizer_gae.step()
            
            results['loss_img_bce'].append(loss_img_bce.cpu().data[0])

            if i % 10 == 0:
                print("Optimization: Epoch{0}/Iter{1}".format(epoch, i),
                      "BCE loss: {:.5f}".format(loss_img_bce.cpu().data[0]))

        return results

    def _joint_eval(self, num_classes):
        #pdb.set_trace()
        var_name = 'self.tag_index_' + str(int(num_classes)) + '_in_1006'
        self.VAE.eval()
        self.GAE.eval()
        gt = np.zeros((0, num_classes))
        pred = np.zeros((0, num_classes))
        for i, (image_fea_, label_, _) in enumerate(self.test_data_loader):
            batchsize = image_fea_.size(0)
            print("{} test images are evaluated.".format(batchsize*i))
            label_ = label_.numpy() #N*1006
            # transfer 1006-dim label to 81-dim label
            label_ = label_[:, eval(var_name)] #N*81
            gt = np.vstack((gt, label_))
            
            _, graph_emb, _ = self.GAE(self.fea) #graph_emb: (1006*128)
            image_fea_ = self._get_variable(image_fea_) #N*2048
            _, image_emb, _ = self.VAE(image_fea_) #N*128
            cls_pred = torch.mm(image_emb, graph_emb.t()) #N*1006
            cls_pred = F.sigmoid(cls_pred)
            cls_pred = cls_pred.data.cpu().numpy()
            cls_pred = cls_pred[:, eval(var_name)]
            pred = np.vstack((pred, cls_pred))
        
        precision_all, recall_all, f1_score_all = self._eval(gt, pred)
        return precision_all, recall_all, f1_score_all


    def _extract(self):
        #pdb.set_trace()
        for i, (images_, paths_) in enumerate(self.data_loader):
            #pdb.set_trace()
            print(i*images_.shape[0])
            images_ = self._get_variable(images_)
            img_embs = self._extract_features(images_)
            img_embs = img_embs.cpu().data.numpy() #N*2048
            for (img_emb_, path_) in zip(img_embs, paths_):
                fea_path = '.'.join((path_.split('.')[0], path_.split('.')[1], 'fea'))
                with open(fea_path, 'wb') as f:
                    pickle.dump(img_emb_, f, pickle.HIGHEST_PROTOCOL)


    def _cls_train(self, epoch, results):
        print("Optimization CLS: Epoch----{0}".format(epoch))
        self.CLS.train()
        pdb.set_trace()

        for i, (image_emb, label_, weight_) in enumerate(self.data_loader):
            #pdb.set_trace()
            #image_ = self._get_variable(image_)
            batchsize = image_emb.shape[0]
            label_ = self._get_variable(label_)
            #img_emb = self._extract_features(image_)
            img_emb = self._get_variable(image_emb)
            weight_ = self._get_variable(weight_)
            #self.CLS.zero_grad()
            self.optimizer_cls.zero_grad()

            #pdb.set_trace()
            cls_predict = self.CLS(img_emb)
            loss_img_bce = F.binary_cross_entropy(cls_predict, label_, weight_, size_average=True)
            loss_img_bce = loss_img_bce / 81.0 * 1006.0
            loss_img_bce.backward()

            self.optimizer_cls.step()
        
            results['loss_img_bce'].append(loss_img_bce.cpu().data[0])

            if i % 10 == 0:
                print("Optimization CLS: Epoch{0}/Iter{1}".format(epoch, i),
                      "Pos loss: {:.5f}".format(loss_img_bce.cpu().data[0]),
                      "Neg loss: {:.5f}".format(loss_img_bce.cpu().data[0]))

        return results
    
    def _cls_eval(self, num_classes):
        self.CLS.eval()
        gt = np.zeros((0, num_classes))
        pred = np.zeros((0, num_classes))
        for i, (image_emb, label_, _) in enumerate(self.test_data_loader):
            batchsize = image_emb.size(0)
            print("{} test images are evaluated.".format(batchsize*i))
            label_ = label_.numpy() #N*1006
            label_ = label_[:, self.tag_index_81_in_1006] #N*81

            gt = np.vstack((gt, label_))
            img_emb = self._get_variable(image_emb)
            #img_emb = self._extract_features(image_)
            cls_pred = self.CLS(img_emb)
            #cls_pred = F.softmax(cls_pred) #1*1006
            cls_pred = cls_pred.data.cpu().numpy()
            cls_pred = cls_pred[:, self.tag_index_81_in_1006]
            pred = np.vstack((pred, cls_pred))

        precision_all, recall_all, f1_score_all = self._eval(gt, pred)
        return precision_all, recall_all, f1_score_all

    def _eval(self, gt, pred):
        num_imgs = pred.shape[0]
        num_classes = pred.shape[1]
        precision_per_image_all = np.zeros((num_imgs))
        recall_per_image_all = np.zeros((num_imgs))
        f1_score_per_image_all = np.zeros((num_imgs))

        precision_per_label_all = np.zeros((num_classes))
        corr_per_label_all = np.zeros((num_classes))
        pred_per_label_all = np.zeros((num_classes))

        k = 3
        upper_bound = np.zeros((num_imgs))
        for i in range(num_imgs):
            #pdb.set_trace()
            single_img_pred = pred[i]
            single_img_gt = gt[i]

            if single_img_gt.sum() > k:
                upper_bound[i] = 1.0
            else:
                upper_bound[i] = float(single_img_gt.sum()) / float(k)


            topk_ind = np.argsort(-single_img_pred)[:k]
            topk_pred = single_img_pred[topk_ind]
            topk_gt = single_img_gt[topk_ind]

            for pre_ind in topk_ind:
                pred_per_label_all[pre_ind] += 1.0
                if single_img_gt[pre_ind] == 1.0:
                    corr_per_label_all[pre_ind] += 1.0

            corr_anno_label = topk_gt.sum()
            precision = float(corr_anno_label) / float(k)
            recall = float(corr_anno_label) / float(single_img_gt.sum())
            if (precision + recall) == 0.0:
                f1_score = 0.0
            else:
                f1_score = 2*precision*recall / (precision + recall)
            precision_per_image_all[i] = precision
            recall_per_image_all[i] = recall
            f1_score_per_image_all[i] = f1_score

        for i in range(num_classes):
            if pred_per_label_all[i] == 0:
                precision_per_label_all[i] = 0.0
            else:
                precision_per_label_all[i] = corr_per_label_all[i] / pred_per_label_all[i]
        
        print("Precison@3 perLabel: {}".format(precision_per_label_all.mean()))
        #pdb.set_trace()
        print("Precision@3 perImage_upper_bound: {}".format(upper_bound.mean()))
        # pred: (N, num_classes)
        # gt: (N, num_classes)
        return precision_per_image_all, recall_per_image_all, f1_score_per_image_all


    def load_model(self):
        print("Load models from {}...".format(self.load_path))

        if self.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        filename = '{}/model.pth'.format(self.load_path)
        '''
        self.model.load_state_dict(torch.load(filename, map_location=map_location))
        '''

    def train(self):

        pdb.set_trace()
        #optimizer = torch.optim.Adam
        #optimizer_gae = optimizer(self.GAE.parameters(), lr=0.01, betas=(0.95, 0.999))
        #optimizer_vae = optimizer(self.VAE.parameters(), lr=0.001, betas=(0.95, 0.9999))

        
        results = defaultdict(list)
        # First optimize the GAE
        '''
        for epoch in range(200):
            results = self._gae_train(epoch, results)
        print("Optimizatoin of the GAE is done.")
        '''

        #self._extract()
        for epoch in range(200):
            results = self._joint_train(epoch, results)
            #results = self._cls_train(epoch, results)
            #evaluation on test images
            precision_all, recall_all, f1_score_all = self._joint_eval(81)
            #precision_all, recall_all, f1_score_all = self._cls_eval(81)

            results['precision'].append(precision_all.mean())
            results['recall'].append(recall_all.mean())
            results['f1_score'].append(f1_score_all.mean())

            print("Evaluation is done.")
            print("One Epoch is done.\n",
                  "Precision = {} (threshold=0.95)\n".format(precision_all.mean()),
                  "Recall = {}\n".format(recall_all.mean()),
                  "F1_score = {}\n".format(f1_score_all.mean()))


        pdb.set_trace()
        with open('./dataset/Results.pkl', 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        '''
        for step in range(self.start_step, self.max_step):
            try:
                img, label = loader.next()
            except StopIteration:
                # Evaluation on test images
                accuracy, roc_score, ap_score = self._vae_eval(1006)
                pdb.set_trace()
                print("Evaluation is done.")
                print("One Epoch is done.\n",
                      "Accuracy = {} (threshold=0.5)\n".format(accuracy.mean()),
                      "ROC_Score = {}\n".format(roc_score.mean()),
                      "AP_Score = {}\n".format(ap_score.mean()))

                print("Continue the training process.")

                loader = iter(self.data_loader)
                img, label = loader.next()
            
            img = self._get_variable(img)
            label = self._get_variable(label)

            img_emb = self._extract_features(img)
            self.VAE.zero_grad()
            recon_emb, vae_mu, vae_std = self.VAE(img_emb)

            # Reconstruction + KL divergence losses
            vae_mse = F.mse_loss(recon_emb, img_emb, size_average=True)
            vae_kld = -(0.5 / batchsize) * torch.mean(torch.sum(1 + 2*torch.log(vae_std) - vae_mu.pow(2) - vae_std.pow(2), 1))
            loss_vae = vae_mse + vae_kld

            pdb.set_trace()
            # classification loss
            self.GAE.eval()
            _, graph_emb, _ = self.GAE(fea, adj_norm) # 1006*300
            cls_predict = torch.mm(vae_mu, graph_emb.detach().t()) # N*300 mul 300*1006
            cls_predict = F.sigmoid(cls_predict)
            loss_img_bce = F.binary_cross_entropy(cls_predict, label, size_average=Tr
            print("xxxx")
        '''
