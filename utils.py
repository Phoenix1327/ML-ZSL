from __future__ import print_function

import os
import json
import logging
import numpy as np
from datetime import datetime

import pickle as pkl
import cPickle
import scipy.sparse as sp
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

import pdb


def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    #pdb.set_trace()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.dataset, get_time())

    if not hasattr(config, 'model_dir'):
        config.model_dir = os.path.join(config.log_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir, config.dataset)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        
        
######################################################################
# ------------------------------------
# Some functions borrowed from:
# https://github.com/tkipf/pygcn and 
# https://github.com/tkipf/gae
# ------------------------------------


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def eval_gae(edges_pos, edges_neg, emb, adj_orig):

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    emb = emb.data.numpy()
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []

    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []

    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])

    accuracy = accuracy_score((preds_all > 0.5).astype(float), labels_all)
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return accuracy, roc_score, ap_score


def make_sparse(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(
            pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


# Subsample sparse variables
def get_subsampler(cuda, variable):
    if cuda:
        variable = variable.cpu()
    data = variable.view(1, -1).data.numpy()
    edges = np.where(data == 1)[1]
    nonedges = np.where(data == 0)[1]

    idx = np.random.choice(
            nonedges.shape[0], edges.shape[0], replace=False)
    
    return np.append(edges, nonedges[idx])

def get_possampler(cuda, variable):
    if cuda:
        variable = variable.cpu()
    data = variable.view(1, -1).data.numpy()
    edges = np.where(data == 1)[1]

    return edges


def plot_results(results, test_freq, path='results.png'):
    # Init
    plt.close('all')
    fig = plt.figure(figsize=(8, 8))

    x_axis_train = range(len(results['train_elbo']))
    x_axis_test = range(0, len(x_axis_train), test_freq)
    # Elbo
    ax = fig.add_subplot(2, 2, 1)
    ax.plot(x_axis_train, results['train_elbo'])
    ax.set_ylabel('Loss (ELBO)')
    ax.set_title('Loss (ELBO)')
    ax.legend(['Train'], loc='upper right')

    # Accuracy
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(x_axis_train, results['accuracy_train'])
    ax.plot(x_axis_test, results['accuracy_test'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend(['Train', 'Test'], loc='lower right')

    # ROC
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(x_axis_train, results['roc_train'])
    ax.plot(x_axis_test, results['roc_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ROC AUC')
    ax.set_title('ROC AUC')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Precision
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(x_axis_train, results['ap_train'])
    ax.plot(x_axis_test, results['ap_test'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Precision')
    ax.legend(['Train', 'Test'], loc='lower right')

    # Save
    fig.tight_layout()
    fig.savefig(path)
