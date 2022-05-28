import copy

import numpy as np
import torch
import gym
from obs_filter import MeanStdFilter
from models import LinearActor, NeuralActor, NeuralActor_NoLayerNorm
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import imageio
from cv2 import cv2

def save_trajs(frames, path, algo_name, seed, noise):
    for i in range(NUM_PER_SEED):
        frame_num = min(200, len(frames[i]))
        file_path = path + '/' + algo_name + '/' + str(seed) + '/' + str(noise) + '/' + str(i)
        if not (os.path.exists(file_path)):
            os.makedirs(file_path)
        for j in range(frame_num):
            img = frames[i][j][..., ::-1]
            imageio.imsave(file_path + '/' + str(j) + '.jpg', img)
        img_list = frames[i][:frame_num]
        imageio.mimsave(file_path + '200_steps.gif', img_list, fps=30)


def do_rollouts(env, policy, obs_filter, obs_noise=0., para_noise=0., render=False):
    avg_step, avg_ret = 0, 0
    policy_pert = copy.deepcopy(policy)
    para = policy.get_flat_param()
    state_dim = policy.input_dim
    obs_bool = bool(obs_noise)
    para_bool = bool(para_noise)
    total_env_frames = []
    for i in range(NUM_PER_SEED):
        if para_bool is True:
            para_pert = para + np.random.normal(0, para_noise, len(para))
            policy_pert.set_flat_param(para_pert)
        state = env.reset()
        is_done = False
        traj_step = 0
        traj_reward = 0
        env_frames = []
        while not is_done and traj_step < 1000:
            if render is True:
                env_frames.append(env.render(mode='rgb_array'))
            obs = obs_filter.forward(torch.FloatTensor(state))
            if obs_bool is True:
                obs += obs_noise * torch.randn(state_dim)
            action = policy_pert.get_action(obs)
            state, reward, is_done, _ = env.step(action)
            traj_reward += reward
            traj_step += 1
        total_env_frames.append(env_frames)
        avg_step += traj_step / NUM_PER_SEED
        avg_ret += traj_reward / NUM_PER_SEED
    return avg_step, avg_ret, total_env_frames


def run_exp(env_name):
    print(env_name_list[env_name])
    env = gym.make(env_name_list[env_name])
    env.seed(SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    algo_name = 'zoac_mat'
    file_path = './learnedpolicies/' + env_name + '/' + algo_name
    for seed in seed_list:
        policy = LinearActor(state_dim, action_dim, SEED)
        obs_filter = MeanStdFilter(state_dim)
        para = np.load(file_path + '/' + str(seed) + '/policy.npy')
        policy.set_flat_param(para)
        mean = torch.load(file_path + '/' + str(seed) + '/mean')
        std = torch.load(file_path + '/' + str(seed) + '/std')
        obs_filter.set(mean, std)
        policy_list[algo_name][str(seed)] = policy
        obs_filter_list[algo_name][str(seed)] = obs_filter
    algo_name = 'zoac_mlp'
    file_path = './learnedpolicies/' + env_name + '/' + algo_name
    for seed in seed_list:
        policy = NeuralActor(state_dim, action_dim, 64, max_action, SEED)
        para = torch.load(file_path + '/' + str(seed) + '/policy.pth')
        policy.load_state_dict(para)
        obs_filter = MeanStdFilter(state_dim)
        mean = torch.load(file_path + '/' + str(seed) + '/mean')
        std = torch.load(file_path + '/' + str(seed) + '/std')
        obs_filter.set(mean, std)
        policy_list[algo_name][str(seed)] = policy
        obs_filter_list[algo_name][str(seed)] = obs_filter
    algo_name = 'ars'
    file_path = './learnedpolicies/' + env_name + '/' + algo_name
    for seed in seed_list:
        policy = LinearActor(state_dim, action_dim, SEED)
        para = np.load(file_path + '/' + str(seed) + '/lin_policy_plus.npz', allow_pickle=True)['arr_0']
        M = torch.FloatTensor(para[0])
        policy.para_weight = M
        obs_filter = MeanStdFilter(state_dim)
        mean = para[1]
        std = para[2]
        obs_filter.set(mean, std)
        policy_list[algo_name][str(seed)] = policy
        obs_filter_list[algo_name][str(seed)] = obs_filter
    for algo_name in ['es', 'ppo']:
        file_path = './learnedpolicies/' + env_name + '/' + algo_name
        for seed in seed_list:
            policy = NeuralActor_NoLayerNorm(state_dim, action_dim, 64, max_action, SEED)
            para = torch.load(file_path + '/' + str(seed) + '/policy.pth')
            policy.load_state_dict(para)
            obs_filter = MeanStdFilter(state_dim)
            mean = torch.load(file_path + '/' + str(seed) + '/mean')
            std = torch.load(file_path + '/' + str(seed) + '/std')
            obs_filter.set(mean, std)
            policy_list[algo_name][str(seed)] = policy
            obs_filter_list[algo_name][str(seed)] = obs_filter
    # run_robust_exp
    for algo_name in algo_name_list:
        print(algo_name_list[algo_name])
        for seed in seed_list:
            print(seed)
            policy = policy_list[algo_name][str(seed)]
            obs_filter = obs_filter_list[algo_name][str(seed)]
            # obs_noise
            obs_return_list = []
            obs_step_list = []
            for obs_noise in obs_noise_list:
                avg_step, avg_ret, frames = do_rollouts(env, policy, obs_filter, obs_noise, 0.)
                obs_return_list.append(avg_ret)
                obs_step_list.append(avg_step)
                if len(frames) > 0:
                    file_path = './final_final/final_learned_policies/nips_result/robust/' + env_name + '/trajs/obs_noise'
                    save_trajs(frames, file_path, algo_name, seed, obs_noise)
            obs_data = np.column_stack((obs_noise_list.tolist(), obs_step_list, obs_return_list))
            obs_df = pd.DataFrame(obs_data, columns=['NoiseStd', 'AvgStep', 'AvgReturn'])
            obs_df.insert(len(obs_df.columns), "Condition", algo_name_list[algo_name])
            obs_df.insert(len(obs_df.columns), "Seed", str(seed))
            obs_noise_result[env_name].append(obs_df)
            # para_noise
            para_return_list = []
            para_step_list = []
            for para_noise in para_noise_list:
                avg_step, avg_ret, frames = do_rollouts(env, policy, obs_filter, 0., para_noise)
                para_return_list.append(avg_ret)
                para_step_list.append(avg_step)
                if len(frames) > 0:
                    file_path = './final_final/final_learned_policies/nips_result/robust/' + env_name + '/trajs/para_noise'
                    save_trajs(frames, file_path, algo_name, seed, para_noise)
            para_data = np.column_stack((para_noise_list.tolist(), para_step_list, para_return_list))
            para_df = pd.DataFrame(para_data, columns=['NoiseStd', 'AvgStep', 'AvgReturn'])
            para_df.insert(len(para_df.columns), "Condition", algo_name_list[algo_name])
            para_df.insert(len(para_df.columns), "Seed", str(seed))
            para_noise_result[env_name].append(para_df)
    obs_noise_result[env_name] = pd.concat(obs_noise_result[env_name], ignore_index=True)
    para_noise_result[env_name] = pd.concat(para_noise_result[env_name], ignore_index=True)
    obs_file_path = './learnedpolicies/' + env_name + '/obs_noise_robustness.csv'
    obs_noise_result[env_name].to_csv(obs_file_path, index=False)
    para_file_path = './learnedpolicies/' + env_name + '/para_noise_robustness.csv'
    para_noise_result[env_name].to_csv(para_file_path, index=False)


def draw_fig(env_name):
    file_path = './learnedpolicies/' + env_name + '/obs_noise_robustness.csv'
    obs_data = pd.read_csv(file_path)
    file_path = './learnedpolicies/' + env_name + '/para_noise_robustness.csv'
    para_data = pd.read_csv(file_path)
    sns.set(style='whitegrid', font_scale=1)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    fig.suptitle(env_name_list[env_name])
    sns.lineplot(ax=axes[0], data=obs_data, x="NoiseStd", y="AvgReturn", hue="Condition", estimator="mean",
                 err_style="band", ci=95)
    axes[0].set_xlim(0, 0.3)
    axes[0].set_ylabel("Total Average Return")
    axes[0].set_xlabel("Noise Scale $\sigma$")
    axes[0].set_title("Robustness to Observation Noise")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles[1:], labels=labels[1:], loc='lower left')
    sns.lineplot(ax=axes[1], data=para_data, x="NoiseStd", y="AvgReturn", hue="Condition", estimator="mean",
                 err_style="band", ci=95, legend=False)
    axes[1].set_xlim(0, 0.1)
    axes[1].set_ylabel(None)
    axes[1].set_xlabel("Noise Scale $\sigma$")
    axes[1].set_title("Robustness to Parameter Noise")
    plt.tight_layout()
    plt.savefig(fname='./learnedpolicies/' + env_name + '/robust_test.svg', format="svg")


SEED = 12345
NUM_PER_SEED = 20
np.random.seed(SEED)
torch.manual_seed(SEED)

env_name_list = {'invdp': 'InvertedDoublePendulum-v2',
                 'hopper': 'Hopper-v2',
                 'halfcheetah': 'HalfCheetah-v2',
                 'ant': 'Ant-v2',
                 'humanstand': 'HumanoidStandup-v2'}
obs_noise_result = {'invdp': [],
                    'hopper': [],
                    'halfcheetah': [],
                    'ant': [],
                    'humanstand': []}
para_noise_result = {'invdp': [],
                     'hopper': [],
                     'halfcheetah': [],
                     'ant': [],
                     'humanstand': []}
algo_name_list = {'zoac_mat': 'ZOAC (Linear)',
                  'zoac_mlp': 'ZOAC (Neural)',
                  'ars': 'ARS (Linear)',
                  'es-clip': 'ES (Neural)',
                  'ppo-clip': 'PPO (Neural)'}
policy_list = {'zoac_mat': {},
               'zoac_mlp': {},
               'ars': {},
               'es-clip': {},
               'ppo-clip': {}}
obs_filter_list = {'zoac_mat': {},
                   'zoac_mlp': {},
                   'ars': {},
                   'es-clip': {},
                   'ppo-clip': {}}

#  plot figure
obs_noise_list = np.arange(0, 0.32, 0.02)
para_noise_list = np.arange(0, 0.11, 0.01)
#  visualize trajectories
# obs_noise_list = np.arange(0, 0.4, 0.1)
# para_noise_list = np.arange(0, 0.14, 0.04)
seed_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for env in ['halfcheetah', 'hopper']:
    run_exp(env)
    draw_fig(env)
