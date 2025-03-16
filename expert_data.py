import os
import numpy as np
import copy
import torch
from config import config
from Astar import AStar
from generate_state import generate_state
from map_3D import generate_map, render_map
from utils import to_one_hot
from replaybuffer_expert import ReplayBufferExpert
import matplotlib.pyplot as plt

# torch.manual_seed(config.seed_number)
# np.random.seed(config.seed_number)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def step(s, s_):
    if s[0] + 1 == s_[0] and s[1] == s_[1] and s[2] == s_[2]:
        return 0
    elif s[0] - 1 == s_[0] and s[1] == s_[1] and s[2] == s_[2]:
        return 1
    elif s[0] == s_[0] and s[1] + 1 == s_[1] and s[2] == s_[2]:
        return 2
    elif s[0] == s_[0] and s[1] - 1 == s_[1] and s[2] == s_[2]:
        return 3
    elif s[0] + 1 == s_[0] and s[1] + 1 == s_[1] and s[2] == s_[2]:
        return 4
    elif s[0] + 1 == s_[0] and s[1] - 1 == s_[1] and s[2] == s_[2]:
        return 5
    elif s[0] - 1 == s_[0] and s[1] + 1 == s_[1] and s[2] == s_[2]:
        return 6
    elif s[0] - 1 == s_[0] and s[1] - 1 == s_[1] and s[2] == s_[2]:
        return 7
    elif s[0] == s_[0] and s[1] == s_[1] and s[2] + 1 == s_[2]:
        return 8
    elif s[0] == s_[0] and s[1] == s_[1] and s[2] - 1 == s_[2]:
        return 9


def select_expert_data(expert_replay_buffer):
    flg = True
    while flg == True:
        map, start, end = generate_map(config.vis_map_size,
                                       obs_num=config.obs_num,
                                       obs_size_min=config.obs_size_min,
                                       obs_size_max=config.obs_size_max,
                                       is_random=config.is_random)
        ins_map = copy.deepcopy(map)
        ins_start = copy.deepcopy(start)
        ins_end = copy.deepcopy(end)

        # 获取专家数据
        try:
            paths, _ = AStar((ins_start[0], ins_start[1], ins_start[2]),
                             (ins_end[0], ins_end[1], ins_end[2]),
                             ins_map, "euclidean").searching()
        except:
            print('A* map is error, skip this loop')
            continue

        path = []
        for poi in paths:
            path.insert(0, poi)
        done = False
        per_act_one_hot = [[0] * 10]
        for i in range(0, len(path) - 1):
            # print(path[i][0], path[i][1], path[i][2])

            state = generate_state(ins_map, [path[i][0], path[i][1], path[i][2]],
                                   [ins_end[0], ins_end[1], ins_end[2]])
            if i == 0:
                per_act_one_hot = [[0] * 10]
                # act_one_hot = to_one_hot(step(path[i], path[i+1]))

            else:
                per_act_one_hot = np.expand_dims(to_one_hot(step(path[i - 1], path[i])), 0)
                # act_one_hot = to_one_hot(step(path[i], path[i + 1]))
            act_one_hot = to_one_hot(step(path[i], path[i + 1]))
            ins_map[path[i][0], path[i][1], path[i][2]] = 0
            ins_map[path[i + 1][0], path[i + 1][1], path[i + 1][2]] = 0.5
            state_ = generate_state(ins_map, [path[i + 1][0], path[i + 1][1], path[i + 1][2]],
                                    [ins_end[0], ins_end[1], ins_end[2]])

            position_t = np.asarray([[path[i][0] - config.boundary,
                                      path[i][1] - config.boundary,
                                      path[i][2] - config.boundary,
                                      ins_end[0] - config.boundary,
                                      ins_end[1] - config.boundary,
                                      ins_end[2] - config.boundary]]) / config.vis_map_size[0]

            position = np.concatenate((position_t, per_act_one_hot), axis=1)

            position_t_ = np.asarray([[path[i + 1][0] - config.boundary,
                                       path[i + 1][1] - config.boundary,
                                       path[i + 1][2] - config.boundary,
                                       ins_end[0] - config.boundary,
                                       ins_end[1] - config.boundary,
                                       ins_end[2] - config.boundary]]) / config.vis_map_size[0]
            # print(position_t_.shape)
            # print(act_one_hot.shape)
            # print(np.expand_dims(act_one_hot, 0).shape)
            # print("---------------")
            position_ = np.concatenate((position_t_, np.expand_dims(act_one_hot, 0)), axis=1)

            if path[i + 1][0] == ins_end[0] and path[i + 1][1] == ins_end[1] and path[i + 1][2] == ins_end[2]:
                done = True
            expert_replay_buffer.store(state, state_, position[0], position_[0], act_one_hot, done)

            if expert_replay_buffer.count == config.batch_size:
                print("获取专家数据数量：{}".format(expert_replay_buffer.count))
                print("********专家数据获取完毕********")
                expert_replay_buffer.count = 0
                flg = False
                break

# expert_replay_buffer = ReplayBufferExpert(config)
# select_expert_data(expert_replay_buffer)
