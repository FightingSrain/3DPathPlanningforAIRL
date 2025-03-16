from config import config
from generate_state import generate_state
import numpy as np
import torch.nn.functional as F
from utils import reward_caculation
from Astar import AStar
import sys
# import cv2
import torch
import copy



# torch.manual_seed(config.seed_number)
# np.random.seed(config.seed_number)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Env():
    def __init__(self, config, agent):
        self.cur_state = None
        self.pre_state = None
        self.agent = agent

    def reset(self, map, start, end):
        self.cur_state = None
        self.pre_state = None
        # --------------------
        self.map = map
        self.cur = copy.deepcopy(start)
        self.per = self.cur
        self.end = copy.deepcopy(end)
        # --------------------
        self.step_num = 0

    def step(self, action1, position, act_one_hot):
        # self.cur_state = state
        # self.pre_state = self.cur_state
        # reward = 0
        done = False

        self.per = copy.deepcopy(self.cur)

        per_state = generate_state(self.map, self.cur, self.end)
        # with torch.no_grad():
        #     per_reward = -((self.agent.target_discriminator(torch.Tensor(per_state),
        #                                             torch.Tensor(position),
        #                                             torch.Tensor([act_one_hot])) - 1) ** 2).mean().cpu().numpy()
        step_num = 1
        r_temp = 0
        # if action2 == 0:
        #     step_num = 1
        # elif action2 == 1:
        #     step_num = 2
        # -----------------------------------
        if action1 == 0:  # 前
            # 区域可走
            if self.map[self.cur[0] + step_num, self.cur[1], self.cur[2]] == 0 \
                    and self.cur[0] + step_num < config.map_size[0] - config.boundary:

                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0] + step_num, self.cur[1], self.cur[2]] = 0.5
                self.cur[0] += step_num
            # 到达目标点
            elif self.map[self.cur[0] + step_num, self.cur[1], self.cur[2]] == -0.5 \
                    and self.cur[0] + step_num < config.map_size[0] - config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0] + step_num, self.cur[1], self.cur[2]] = 0.5
                self.cur[0] += step_num
                done = True
            elif self.map[self.cur[0] + step_num, self.cur[1], self.cur[2]] == 1 \
                     and self.cur[0] + step_num < config.map_size[0] - config.boundary:
                r_temp -= 1

        elif action1 == 1:  # 后
            # 区域可走
            if self.map[self.cur[0] - step_num, self.cur[1], self.cur[2]] == 0 \
                    and self.cur[0] - step_num >= config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0] - step_num, self.cur[1], self.cur[2]] = 0.5
                self.cur[0] -= step_num
            # 到达目标点
            elif self.map[self.cur[0] - step_num, self.cur[1], self.cur[2]] == -0.5 \
                    and self.cur[0] - step_num >= config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0] - step_num, self.cur[1], self.cur[2]] = 0.5
                self.cur[0] -= step_num
                done = True
            elif self.map[self.cur[0] - step_num, self.cur[1], self.cur[2]] == 1 \
                    and self.cur[0] - step_num >= config.boundary:
                r_temp -= 1

        elif action1 == 2:  # 左
            if self.map[self.cur[0], self.cur[1] + step_num, self.cur[2]] == 0 \
                    and self.cur[1] + step_num < config.map_size[1] - config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0], self.cur[1] + step_num, self.cur[2]] = 0.5
                self.cur[1] += step_num
            elif self.map[self.cur[0], self.cur[1] + step_num, self.cur[2]] == -0.5 \
                    and self.cur[1] + step_num < config.map_size[1] - config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0], self.cur[1] + step_num, self.cur[2]] = 0.5
                self.cur[1] += step_num
                done = True
            elif self.map[self.cur[0], self.cur[1] + step_num, self.cur[2]] == 1 \
                    and self.cur[1] + step_num < config.map_size[1] - config.boundary:
                r_temp -= 1

        elif action1 == 3:  # 右
            if self.map[self.cur[0], self.cur[1] - step_num, self.cur[2]] == 0 \
                    and self.cur[1] - step_num >= config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0], self.cur[1] - step_num, self.cur[2]] = 0.5
                self.cur[1] -= step_num
            elif self.map[self.cur[0], self.cur[1] - step_num, self.cur[2]] == -0.5 \
                    and self.cur[1] - step_num >= config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0], self.cur[1] - step_num, self.cur[2]] = 0.5
                self.cur[1] -= step_num
                done = True
            elif self.map[self.cur[0], self.cur[1] - step_num, self.cur[2]] == 1 \
                    and self.cur[1] - step_num >= config.boundary:
                r_temp -= 1

        elif action1 == 4:  # 前左
            if self.map[self.cur[0] + step_num, self.cur[1] + step_num, self.cur[2]] == 0 \
                and (self.map[self.cur[0] + step_num, self.cur[1], self.cur[2]] != 1 \
                or self.map[self.cur[0], self.cur[1] + step_num, self.cur[2]] != 1) \
                and self.cur[0] + step_num < config.map_size[0] - config.boundary \
                and self.cur[1] + step_num < config.map_size[1] - config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0] + step_num, self.cur[1] + step_num, self.cur[2]] = 0.5
                self.cur[0] += step_num
                self.cur[1] += step_num
            elif self.map[self.cur[0] + step_num, self.cur[1] + step_num, self.cur[2]] == -0.5 \
                    and (self.map[self.cur[0] + step_num, self.cur[1], self.cur[2]] != 1 \
                    or self.map[self.cur[0], self.cur[1] + step_num, self.cur[2]] != 1) \
                    and self.cur[0] + step_num < config.map_size[0] - config.boundary \
                    and self.cur[1] + step_num < config.map_size[1] - config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0] + step_num, self.cur[1] + step_num, self.cur[2]] = 0.5
                self.cur[0] += step_num
                self.cur[1] += step_num
                done = True
            elif self.map[self.cur[0] + step_num, self.cur[1] + step_num, self.cur[2]] == 1 \
                    and self.cur[0] + step_num < config.map_size[0] - config.boundary \
                    and self.cur[1] + step_num < config.map_size[1] - config.boundary:
                r_temp -= 1

        elif action1 == 5:  # 前右
            if self.map[self.cur[0] + step_num, self.cur[1] - step_num, self.cur[2]] == 0 \
                    and (self.map[self.cur[0] + step_num, self.cur[1], self.cur[2]] != 1 \
                    or self.map[self.cur[0], self.cur[1] - step_num, self.cur[2]] != 1) \
                    and self.cur[0] + step_num < config.map_size[0] - config.boundary \
                    and self.cur[1] - step_num >= config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0] + step_num, self.cur[1] - step_num, self.cur[2]] = 0.5
                self.cur[0] += step_num
                self.cur[1] -= step_num
            elif self.map[self.cur[0] + step_num, self.cur[1] - step_num, self.cur[2]] == -0.5 \
                    and (self.map[self.cur[0] + step_num, self.cur[1], self.cur[2]] != 1 \
                    or self.map[self.cur[0], self.cur[1] - step_num, self.cur[2]] != 1) \
                    and self.cur[0] + step_num < config.map_size[0] - config.boundary \
                    and self.cur[1] - step_num >= config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0] + step_num, self.cur[1] - step_num, self.cur[2]] = 0.5
                self.cur[0] += step_num
                self.cur[1] -= step_num
                done = True
            elif self.map[self.cur[0] + step_num, self.cur[1] - step_num, self.cur[2]] == 1 \
                    and self.cur[0] + step_num < config.map_size[0] - config.boundary \
                    and self.cur[1] - step_num >= config.boundary:
                r_temp -= 1

        elif action1 == 6:  # 后左
            if self.map[self.cur[0] - step_num, self.cur[1] + step_num, self.cur[2]] == 0 \
                    and (self.map[self.cur[0] - step_num, self.cur[1], self.cur[2]] != 1 \
                    or self.map[self.cur[0], self.cur[1] + step_num, self.cur[2]] != 1) \
                    and self.cur[0] - step_num >= config.boundary \
                    and self.cur[1] + step_num < config.map_size[1] - config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0] - step_num, self.cur[1] + step_num, self.cur[2]] = 0.5
                self.cur[0] -= step_num
                self.cur[1] += step_num
            elif self.map[self.cur[0] - step_num, self.cur[1] + step_num, self.cur[2]] == -0.5 \
                    and (self.map[self.cur[0] - step_num, self.cur[1], self.cur[2]] != 1 \
                    or self.map[self.cur[0], self.cur[1] + step_num, self.cur[2]] != 1) \
                    and self.cur[0] - step_num >= config.boundary \
                    and self.cur[1] + step_num < config.map_size[1] - config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0] - step_num, self.cur[1] + step_num, self.cur[2]] = 0.5
                self.cur[0] -= step_num
                self.cur[1] += step_num
                done = True
            elif self.map[self.cur[0] - step_num, self.cur[1] + step_num, self.cur[2]] == 1 \
                    and self.cur[0] - step_num >= config.boundary \
                    and self.cur[1] + step_num < config.map_size[1] - config.boundary:
                r_temp -= 1

        elif action1 == 7:  # 后右
            if self.map[self.cur[0] - step_num, self.cur[1] - step_num, self.cur[2]] == 0 \
                    and (self.map[self.cur[0] - step_num, self.cur[1], self.cur[2]] != 1 \
                    or self.map[self.cur[0], self.cur[1] - step_num, self.cur[2]] != 1) \
                    and self.cur[0] - step_num >= config.boundary \
                    and self.cur[1] - step_num >= config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0] - step_num, self.cur[1] - step_num, self.cur[2]] = 0.5
                self.cur[0] -= step_num
                self.cur[1] -= step_num
            elif self.map[self.cur[0] - step_num, self.cur[1] - step_num, self.cur[2]] == -0.5 \
                    and (self.map[self.cur[0] - step_num, self.cur[1], self.cur[2]] != 1 \
                    or self.map[self.cur[0], self.cur[1] - step_num, self.cur[2]] != 1) \
                    and self.cur[0] - step_num >= config.boundary \
                    and self.cur[1] - step_num >= config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0] - step_num, self.cur[1] - step_num, self.cur[2]] = 0.5
                self.cur[0] -= step_num
                self.cur[1] -= step_num
                done = True
            elif self.map[self.cur[0] - step_num, self.cur[1] - step_num, self.cur[2]] == 1 \
                    and self.cur[0] - step_num >= config.boundary \
                    and self.cur[1] - step_num >= config.boundary:
                r_temp -= 1

        elif action1 == 8:  # 上
            if self.map[self.cur[0], self.cur[1], self.cur[2] + step_num] == 0 \
                    and self.cur[2] + step_num < config.map_size[2] - config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0], self.cur[1], self.cur[2] + step_num] = 0.5
                self.cur[2] += step_num
            elif self.map[self.cur[0], self.cur[1], self.cur[2] + step_num] == -0.5 \
                    and self.cur[2] + step_num < config.map_size[2] - config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0], self.cur[1], self.cur[2] + step_num] = 0.5
                self.cur[2] += step_num
                done = True
            elif self.map[self.cur[0], self.cur[1], self.cur[2] + step_num] == 1 \
                    and self.cur[2] + step_num < config.map_size[2] - config.boundary:
                r_temp -= 1

        elif action1 == 9:  # 下
            if self.map[self.cur[0], self.cur[1], self.cur[2] - step_num] == 0 \
                    and self.cur[2] - step_num >= config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0], self.cur[1], self.cur[2] - step_num] = 0.5
                self.cur[2] -= step_num
            elif self.map[self.cur[0], self.cur[1], self.cur[2] - step_num] == -0.5 \
                    and self.cur[2] - step_num >= config.boundary:
                self.map[self.cur[0], self.cur[1], self.cur[2]] = 0
                self.map[self.cur[0], self.cur[1], self.cur[2] - step_num] = 0.5
                self.cur[2] -= step_num
                done = True
            elif self.map[self.cur[0], self.cur[1], self.cur[2] - step_num] == 1 \
                    and self.cur[2] - step_num >= config.boundary:
                r_temp -= 1


        # r_vis = reward_caculation(self.per, self.end) - \
        #          reward_caculation(self.cur, self.end)
        a = 0.995
        b = 1.005
        reward_grad = a * reward_caculation(self.per, self.end) - b * reward_caculation(self.cur, self.end)
        # reward_grad = -reward_caculation(self.cur, self.end)

        reward_vis = reward_caculation(self.per, self.end) - reward_caculation(self.cur, self.end)

        for i in range(3):
            position[:, i] = (self.cur[i] - config.boundary) / config.vis_map_size[0]  # 输入网络归一化
        for i in range(6, 16):
            position[0, i] = act_one_hot[i - 6]
        # update map
        next_state = generate_state(self.map, self.cur, self.end)

        # reward = -((self.agent.discriminator(torch.Tensor(next_state),
        #                                    torch.Tensor(position),
        #                                      torch.Tensor([act_one_hot])) - 1)**2).mean().detach().cpu().numpy()
        # with torch.no_grad():
        #     # reward = -F.logsigmoid(-self.agent.discriminator(torch.Tensor(next_state),
        #     #                                torch.Tensor(position),
        #     #                                  torch.Tensor([act_one_hot]))).cpu().numpy()
        #     cur_reward = -((self.agent.target_discriminator(torch.Tensor(next_state),
        #                              torch.Tensor(position),
        #                              torch.Tensor([act_one_hot])) - 1)**2).mean().cpu().numpy()
        cur_reward = 0
        # reward = r_vis
        # if done is True:
            # reward_grad += 50
            # reward_vis += 50

        self.step_num += 1
        if config.is_use_sample_steps:
            if self.step_num == config.sample_steps:
                done = True
        return next_state, reward_grad, reward_vis, done, \
               copy.deepcopy(self.cur), copy.deepcopy(position)
