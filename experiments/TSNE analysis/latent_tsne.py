from __future__ import print_function, division, absolute_import

import os
import pandas as pd
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import itemfreq
import seaborn as sns

try:
  from MulticoreTSNE import MulticoreTSNE as TSNE
except ImportError:
  from sklearn.manifold import TSNE
# from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

SEED = 5218
np.random.seed(SEED)
# ===========================================================================
# Constants
# ===========================================================================
# ====== loading all the files ====== #
Z_original = np.squeeze(np.load('../../features/full_test_x_encoded.npy'), axis=1)
Z_maximize = np.squeeze(np.load('../../features/full_maximized_test_x_encoded.npy'), axis=1)

y_command = np.load('../../features/full_test_y.npy').astype('int32')
y_speaker = np.load('../../features/full_test_speaker_ids.npy')
y_missed = np.load('../../features/misclassified_encoded_test_samples_indices.npy')
# ===========================================================================
# Helper
# ===========================================================================
def get_indices(speaker_set, command_set):
  return np.array(
      [i
       for i, (s, c) in enumerate(zip(y_speaker, y_command))
       if s in speaker_set and
       c in command_set], dtype='int32')

def get_data(n_cmd, n_spk, only_missed=False):
  if only_missed:
    # most popular MIS-CLASSIFIED command based on utterances count
    top_cmd = itemfreq(y_command[y_missed.astype('int32')])
    top_spk = itemfreq(y_speaker[y_missed.astype('int32')])
  else:
    top_spk = itemfreq(y_speaker)
    top_cmd = itemfreq(y_command)
  top_cmd = top_cmd[np.argsort(top_cmd[:, 1])][::-1]
  top_cmd = top_cmd[:, 0]

  # most speaker command based on utterances count
  top_spk = top_spk[np.argsort(top_spk[:, 1].astype('int32'))][::-1]
  top_spk = top_spk[:, 0]

  spk = top_spk[:n_spk]
  cmd = top_cmd[:n_cmd]
  ids = get_indices(speaker_set=spk, command_set=cmd)
  if only_missed:
    ids = np.array([i for i in ids if i in y_missed],
                   dtype='int32')

  y_cmd = y_command[ids]
  y_spk = y_speaker[ids]

  z_org = Z_original[ids]
  z_max = Z_maximize[ids]

  tsne = TSNE(random_state=SEED)
  t = tsne.fit_transform(np.concatenate((z_org, z_max), axis=0))
  t_org = t[:z_org.shape[0]]
  t_max = t[z_org.shape[0]:]

  return t_org, t_max, y_cmd, y_spk

# ===========================================================================
# First Analysis,
# effect of maximization command
# ===========================================================================
def plot_maximizing_command_differences(n_cmd, n_spk, save_path,
                                        arrow_head=0.25):
  palette = sns.color_palette(n_colors=n_cmd)
  t_org, t_max, y_cmd, y_spk = get_data(n_cmd, n_spk, only_missed=True)
  t = np.concatenate((t_org, t_max), axis=0)
  status = np.array(['Original'] * t_org.shape[0] +
                    ['Maximized'] * t_max.shape[0])

  plt.figure(figsize=(12, 12))

  sns.scatterplot(
      x='x', y='y',
      hue='Command ID',
      style='Status',
      # size=500,
      # size_order=['Maximize', 'Original'],
      alpha=0.66,
      data=pd.DataFrame({
        'x': t[:, 0], 'y': t[:, 1],
        'Command ID': np.concatenate((y_cmd, y_cmd)),
        'Speaker ID': np.concatenate((y_spk, y_spk)),
        'Status': status}),
      palette=palette,
      s=120)
  # ====== draw the arrow ====== #
  max_d = np.max(t) - np.min(t)
  for x_org, x_max in zip(t_org, t_max):
    if np.abs(x_max[0] - x_org[0]) < 0.01 * max_d and \
       np.abs(x_max[1] - x_org[1]) < 0.01 * max_d:
      continue
    plt.arrow(
        x_org[0], x_org[1],
        0.95 * (x_max[0] - x_org[0]),
        0.95 * (x_max[1] - x_org[1]),
        linewidth=1,
        linestyle='--',
        head_width=arrow_head,
        head_length=arrow_head,
        color='red',
        alpha=0.8
    )

  plt.legend(fontsize=16)
  plt.xticks([], []); plt.yticks([], [])
  plt.xlabel(None); plt.ylabel(None)
  plt.gcf().savefig(save_path)

plot_maximizing_command_differences(n_cmd=3, n_spk=80,
                                    arrow_head=0.25,
                                    save_path='maximization_effect.png')

# ===========================================================================
# Speaker effect on the maximization
# ===========================================================================
def plot_speaker_effect(n_cmd, n_spk, save_path,
                        arrow_head=0.25,
                        arrow_percent=1.0):
  palette = sns.color_palette(n_colors=n_spk)
  t_org, t_max, y_cmd, y_spk = get_data(n_cmd, n_spk, only_missed=False)
  t = np.concatenate((t_org, t_max), axis=0)
  status = np.array(['Original'] * t_org.shape[0] +
                    ['Maximized'] * t_max.shape[0])

  plt.figure(figsize=(12, 12))
  sns.scatterplot(
      x='x', y='y',
      hue='Speaker ID',
      style='Status',
      # size=500,
      # size_order=['Maximize', 'Original'],
      alpha=0.66,
      data=pd.DataFrame({
        'x': t[:, 0], 'y': t[:, 1],
        'Command ID': np.concatenate((y_cmd, y_cmd)),
        'Speaker ID': np.concatenate((y_spk, y_spk)),
        'Status': status}),
      palette=palette,
      s=188)
  # ====== add the text of the command ====== #
  for (x, y), c in zip(t_max, y_cmd):
    plt.text(x, y, s='  ' + str(c),
            fontsize=12,
             # horizontalalignment='center',
             # verticalalignment='center'
    )
  # ====== draw the arrow ====== #
  for x_org, x_max in zip(t_org, t_max):
    if np.random.rand() < float(arrow_percent):
      plt.arrow(
          x_org[0], x_org[1],
          0.95 * (x_max[0] - x_org[0]),
          0.95 * (x_max[1] - x_org[1]),
          linewidth=1,
          linestyle='--',
          head_width=arrow_head,
          head_length=arrow_head,
          color='red',
          alpha=0.8
      )

  plt.legend(fontsize=16)
  plt.xticks([], []); plt.yticks([], [])
  plt.xlabel(None); plt.ylabel(None)
  plt.gcf().savefig(save_path)

plot_speaker_effect(n_cmd=2, n_spk=2,
                    arrow_head=0.25,
                    save_path='speaker_effect.png')
