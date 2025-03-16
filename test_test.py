# -*- coding:utf-8 -*-
# @Time : 2022/9/17 9:34
# @Author: hjl
# @File : test_test.py
import os
import pickle

import numpy as np
import copy
import torch
from config import config
from agent import PPO_discrete
from Env import Env
from replaybuffer import ReplayBuffer

from generate_state import generate_state
from map_3D import generate_map, render_map
from utils import rewards_map_random_tst, reward_caculation, distance_caculation, record_inf
from tst import tst


# map, start, end = generate_map(config.vis_map_size,
#                                obs_num=config.obs_num,
#                                obs_size_min=config.obs_size_min,
#                                obs_size_max=config.obs_size_max,
#                                is_random=config.is_random, test=True)
def tst1(agent, env, config, epi, ij, rewards_tst_vis, distance_tst_vis, map, start, end):
    net = agent.actor
    net.eval()
    map, start, end = map, start, end
    map[start[0], start[1], start[2]] = 0.5  # 当前位置
    map[end[0], end[1], end[2]] = -0.5  # 目标位置
    ins_map = copy.deepcopy(map)
    ins_start = copy.deepcopy(start)
    ins_end = copy.deepcopy(end)
    env.reset(ins_map, ins_start, ins_end)

def tst(agent, env, config, epi, ij, rewards_tst_vis, distance_tst_vis):

    net = agent.actor
    net.eval()
    map, start, end = generate_map(config.vis_map_size,
                                   obs_num=config.obs_num,
                                   obs_size_min=config.obs_size_min,
                                   obs_size_max=config.obs_size_max,
                                   is_random=config.is_random)
    map[start[0], start[1], start[2]] = 0.5  # 当前位置
    map[end[0], end[1], end[2]] = -0.5  # 目标位置
    ins_map = copy.deepcopy(map)
    ins_start = copy.deepcopy(start)
    ins_end = copy.deepcopy(end)
    env.reset(ins_map, ins_start, ins_end)

    ins_state = generate_state(ins_map, ins_start, ins_end)
    ins_position = np.asarray([[ins_start[0] - config.boundary,
                                ins_start[1] - config.boundary,
                                ins_start[2] - config.boundary,
                                ins_end[0] - config.boundary,
                                ins_end[1] - config.boundary,
                                ins_end[2] - config.boundary]]) / config.vis_map_size[0]

    done = False
    # h_in = np.zeros((9, 1, 1, 32))
    h_in = np.zeros((1, 1, 64))
    state = copy.deepcopy(ins_state)  # ins_state
    position = copy.deepcopy(ins_position)

    target_tst_reward = reward_caculation(ins_start, ins_end)  # , target=True

    posx = [ins_start[0] + 0.5 - config.boundary]
    posy = [ins_start[1] + 0.5 - config.boundary]
    posz = [ins_start[2] + 0.5 - config.boundary]

    sum_step = 0

    rewards_tst_temp = []
    distance_tst_temp = []
    while not done:
        sum_step += 1
        action, a_logprob,  v, h_out = agent.choose_action(state, position, h_in, test=True)
        state_, r_grad, r_vis, done, cur_pos, position_ = env.step(action, position)
        # print('action={},cur={},r={}'.format(action,cur_pos,r))

        # if epi > 5000:
        #     inf = (math.exp(a_logprob2), action2)
        #     with open("./figure/random/information/a_logprob2_tst.csv", "ab") as f:
        #         np.savetxt(f, inf)

        # for i in range(3):
        #     position[:, i] = (cur_pos[i] - config.boundary) / config.vis_map_size[0]  # 输入网络归一化
        _, _, v_, _ = agent.choose_action(state_, position_, h_out)
        posx.append(cur_pos[0] + 0.5 - config.boundary)
        posy.append(cur_pos[1] + 0.5 - config.boundary)
        posz.append(cur_pos[2] + 0.5 - config.boundary)

        state = state_
        h_in = h_out
        position = position_

        rewards_tst_temp.append(r_vis)
        # if done and sum_step != config.sample_steps:
        #     r -= 50
        #     rewards_tst_abs.append(abs(r))
        # else:
        #     rewards_tst_abs.append(abs(r))


    rewards_tst_vis.append(np.sum(rewards_tst_temp) / target_tst_reward)
    distance_tst_temp = distance_caculation(posx, posy, posz)
    distance_tst_vis.append(distance_tst_temp / (target_tst_reward-50))
    # if epi >= 1000:
    #     render_map(map, posx, posy, posz, ins_start, ins_end)

    if epi % 10 == 0 and ij == config.tst_pool-1:
        rewards_map_random_tst(1, epi, rewards_tst_vis, distance_tst_vis)

    if epi >= 0:
        if rewards_tst_vis[-1] == 1.0 or cur_pos == ins_end:
            is_which = 1
            successful = 1
        elif rewards_tst_vis[-1] == 1.0 and distance_tst_vis[-1] == 1.0:
            is_which = 2
            successful = 0
        elif cur_pos != ins_end:
            is_which = 3
            successful = 0

        record_inf("0.95", config.seed_number, is_which, epi, ij, map, posx, posy, posz, ins_start, ins_end)

    # succ = 0
    # if rewards_tst_vis[-1] == 1.0:
    #     succ += 1
    # if epi % 500 == 0:
    #     success(epi, succ)
    net.train()
    return successful


def record_inf(ab, seed, is_which, epi, ij, map, posx, posy, posz, start, end):
    path_1 = "./figure/random.tst/information/"
    if is_which == 1:
        path_2 = "reward_succ"
    elif is_which == 3:
        path_2 = "fail"
    path_map3 = "/{0}-{1}-map.pkl".format(str(epi), str(ij))
    path_other3 = "/{0}-{1}-other.pkl".format(str(epi), str(ij))
    map_path = path_1 + "{}/{}/".format(ab, "seed"+str(seed)) + path_2 + path_map3
    other_path = path_1 + "{}/{}/".format(ab, "seed"+str(seed)) + path_2 + path_other3


    with open(map_path, "wb") as f:
        pickle.dump(map, f)

    other = [posx] + [posy] + [posz] + [start] + [end]
    # print(other)
    other = np.array(other, dtype=object)
    # print(other.shape)

    with open(other_path, "wb") as f:
        pickle.dump(other, f)



model_path = 'figure/random.tst/model_test_endR/modela_20000-0.95.pth'
agent = PPO_discrete(config)
env_test = Env(config)

net = agent.actor
net.load_state_dict(torch.load(model_path), strict=True)
net.eval()

rewards_tst_vis = []
distance_tst_vis = []
succ = 0
for epi in range(1, 101):
    for ij in range(10):
        # epi = 2700
        # ij = 3
        print('test-{}-{}'.format(epi, ij))
        # map, posx, posy, posz, start, end = print_inf(epi, ij)
        # print('start={},end={}'.format(start,end))
        succ += tst(agent, env_test, config, epi, ij, rewards_tst_vis, distance_tst_vis)
    print("成功次数：{}".format(succ))

print("1000次概率：{}".format(float(succ/1000)))
        # tst1(agent, env_test, config, epi, ij, rewards_tst_vis, distance_tst_vis, map, start, end)

