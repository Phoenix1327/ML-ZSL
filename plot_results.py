from __future__ import print_function

import os
import json
import numpy as np
import pdb

import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


name = 'AIdentity'
pkl_path = './dataset/Results_' + name + '.pkl'
jpg_path = './dataset/Results_' + name + '.jpg'

with open(pkl_path, 'rb') as f:
    results = cPickle.load(f)

pdb.set_trace()

plt.close('all')
fig = plt.figure(figsize=(8, 8))

x_axis_loss = range(len(results['loss_img_bce']))
x_axis_eval = range(len(results['precision']))

# loss
ax = fig.add_subplot(2, 2, 1)
ax.plot(x_axis_loss, results['loss_img_bce'])
ax.set_ylabel('Loss')
ax.set_title('Loss')
ax.legend(['Train'], loc='upper right')

# Accuracy
ax = fig.add_subplot(2, 2, 2)
ax.plot(x_axis_eval, results['precision'])
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy')
ax.legend(['Train'], loc='lower right')

# Recall
ax = fig.add_subplot(2, 2, 3)
ax.plot(x_axis_eval, results['recall'])
ax.set_ylabel('Recall')
ax.set_title('Recall')
ax.legend(['Train'], loc='lower right')

# F1-Score
ax = fig.add_subplot(2, 2, 4)
ax.plot(x_axis_eval, results['f1_score'])
ax.set_ylabel('F1-Score')
ax.set_title('F1-Score')
ax.legend(['Test'], loc='lower right')

# Save
fig.tight_layout()
fig.savefig(jpg_path)
