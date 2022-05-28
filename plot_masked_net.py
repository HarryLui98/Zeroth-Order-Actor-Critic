import copy

import numpy as np
import torch
import gym
from obs_filter import MeanStdFilter
from models import LinearActor, NeuralActor, NeuralActor_NoLayerNorm, MaskedNeuralActor
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import imageio
from cv2 import cv2

SEED = 12345
env = gym.make("Ant-v2")
env.seed(SEED)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
file_path = "./learnedpolicies/nondiff/policy_epoch_1901.npy"
policy = MaskedNeuralActor(state_dim, action_dim)
para = np.load(file_path)
policy.set_flat_param(para)
para_list = policy.plot_para()
para_l1, para_l2, para_l3 = para_list[1].numpy(), para_list[3].numpy(), para_list[5].numpy()
fig, ax = plt.subplots()
c = ax.pcolor(para_l1, edgecolors='k', linestyle='-', linewidths=0.2, cmap="binary", vmin=0.0, vmax=0.000001)
plt.xlim((0, para_l1.shape[1]))
ax.set_xticks(np.array([0, state_dim-1])+0.5, minor=False)
ax.set_yticks(np.array([0, 63])+0.5, minor=False)
xticklabels = [1, state_dim]
yticklabels = [1, 64]
ax.set_xticklabels(xticklabels, minor=False)
ax.set_yticklabels(yticklabels, minor=False)
ax = plt.gca()
for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
fig = plt.gcf()
fig.set_size_inches((0.05 * state_dim, 0.05 * 64))
plt.tight_layout()
plt.savefig(fname='./learnedpolicies/nondiff/layer1.svg', format='svg')
fig, ax = plt.subplots()
c = ax.pcolor(para_l2, edgecolors='k', linestyle='-', linewidths=0.2, cmap="binary", vmin=0.0, vmax=0.000001)
plt.xlim((0, para_l2.shape[1]))
ax.set_xticks(np.array([0, 63])+0.5, minor=False)
ax.set_yticks(np.array([0, 63])+0.5, minor=False)
xticklabels = [1, 64]
yticklabels = [1, 64]
ax.set_xticklabels(xticklabels, minor=False)
ax.set_yticklabels(yticklabels, minor=False)
ax = plt.gca()
for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
fig = plt.gcf()
fig.set_size_inches((0.05 * 64, 0.05 * 64))
plt.tight_layout()
plt.savefig(fname='./learnedpolicies/nondiff/layer2.svg', format='svg')
fig, ax = plt.subplots()
c = ax.pcolor(para_l3.T, edgecolors='k', linestyle='-', linewidths=0.2, cmap="binary", vmin=0.0, vmax=0.000001)
plt.xlim((0, para_l3.T.shape[1]))
ax.set_xticks(np.array([0, action_dim-1])+0.5, minor=False)
ax.set_yticks(np.array([0, 63])+0.5, minor=False)
xticklabels = [1, action_dim]
yticklabels = [1, 64]
ax.set_xticklabels(xticklabels, minor=False)
ax.set_yticklabels(yticklabels, minor=False)
ax = plt.gca()
for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
fig = plt.gcf()
fig.set_size_inches((0.12 * action_dim, 0.05 * 64))
plt.tight_layout()
plt.savefig(fname='./learnedpolicies/nondiff/layer3.svg', format='svg')