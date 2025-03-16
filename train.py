import os
import numpy as np
import copy
import torch
torch.cuda.current_device()
from config import config
from agent import PPO_discrete
from Env import Env
from replaybuffer import ReplayBuffer
from replaybuffer_expert import ReplayBufferExpert
from expert_data import select_expert_data
from Astar import AStar
from generate_state import generate_state
from map_3D import generate_map, render_map
from utils import rewards_map_random, reward_caculation, distance_caculation
from utils import to_one_hot
from tst import tst
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import math
import matplotlib.pyplot as plt
# torch.manual_seed(config.seed_number)
# np.random.seed(config.seed_number)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def train():
    agent = PPO_discrete(config)
    env = Env(config, agent)
    env_test = Env(config, agent)
    replay_buffer = ReplayBuffer(config)
    expert_replay_buffer = ReplayBufferExpert(config)

    # ----------------------------------------
    total_steps = 0
    epi = 0
    # ---------------------------------------
    rewards_vis = []
    rewards_tst_vis = []
    distance_vis = []
    distance_tst_vis = []
    while total_steps < config.max_train_steps:
        if epi % 1000 == 0 and epi > 0 and agent.entropy_coef > 0.01:
            agent.entropy_coef -= 4e-3
            if agent.entropy_coef < 0.01:
                agent.entropy_coef = 0.01
        map, start, end = generate_map(config.vis_map_size,
                                       obs_num=config.obs_num,
                                       obs_size_min=config.obs_size_min,
                                       obs_size_max=config.obs_size_max,
                                       is_random=config.is_random)
        ins_map = copy.deepcopy(map)
        ins_start = copy.deepcopy(start)
        ins_end = copy.deepcopy(end)

        env.reset(ins_map, ins_start, ins_end)
        target_reward = reward_caculation(ins_start, ins_end)  # , target=True
        # target_reward = 1

        ins_state = generate_state(ins_map, ins_start, ins_end)

        ins_position = np.asarray([[ins_start[0] - config.boundary,
                                    ins_start[1] - config.boundary,
                                    ins_start[2] - config.boundary,
                                    ins_end[0] - config.boundary,
                                    ins_end[1] - config.boundary,
                                    ins_end[2] - config.boundary]]) / config.vis_map_size[0]
        # ==============
        episode_steps = 0
        done = False

        state = copy.deepcopy(ins_state)
        position = copy.deepcopy(ins_position)

        per_action = [[0] * 10]
        position = np.concatenate((position, per_action), axis=1)

        h_in = np.zeros((1, 1, 128))  # 15
        posx = [ins_start[0] + 0.5 - config.boundary]
        posy = [ins_start[1] + 0.5 - config.boundary]
        posz = [ins_start[2] + 0.5 - config.boundary]

        rewards_temp = []
        distance_temp = []


        while not done:
            episode_steps += 1
            # plt.imshow(state[0, 2])
            # plt.show()
            action, a_logprob, v = agent.choose_action(state, position)
            act_one_hot = to_one_hot(action)

            state_, r_grad, r_vis, done, cur_pos, position_ = env.step(action, position, act_one_hot)
            action_, _, v_ = agent.choose_action(state_, position_, target=config.is_target)
            act_one_hot_ = to_one_hot(action_)
            # for i in range(6, 16):
            #     position_[0, i] = act_one_hot[i - 6]
            with torch.no_grad():
                if config.is_target:
                    r2 = -F.logsigmoid(-agent.target_discriminator(torch.Tensor(state_),
                                                            torch.Tensor(position_),
                                                            torch.Tensor([act_one_hot_]), agent.fsn)).cpu().numpy()
                else:
                    r2 = agent.discriminator.cal_reward(torch.Tensor(state),
                                                        torch.Tensor(position),
                                                        torch.Tensor(state_),
                                                        torch.Tensor(position_),
                                                        torch.Tensor([a_logprob]).cuda(),
                                                        config.gamma, done, agent.fsn).cpu().numpy()
            # r = (r2 - r1)*1
            if r2 < 1:
                r2 = np.array([[1e-6]])
            r = r2 + r_grad
            agent.writer.add_scalar(tag='R/TrainLoss', scalar_value=r, global_step=total_steps)
            print('reward={}, r2={}, l2={}'.format(r, r2, r_grad))
            print("TTTTTTTTTTTTT")
            posx.append(cur_pos[0] + 0.5 - config.boundary)
            posy.append(cur_pos[1] + 0.5 - config.boundary)
            posz.append(cur_pos[2] + 0.5 - config.boundary)

            if done is True:
                epi += 1
            if epi % 50 == 0:
                if done is True:
                    render_map(map, posx, posy, posz, ins_start, ins_end)

            # 回合结束 判断是否是到达采样步数上线
            # dw = True 到达目标点
            # dw = False 到达采样步数上限
            if config.is_use_sample_steps:
                if done and episode_steps != config.sample_steps:
                    dw = True
                else:
                    dw = False
            else:
                if done:
                    dw = True
                else:
                    dw = False

            replay_buffer.store(state, action, act_one_hot, act_one_hot_, a_logprob, r, state_,
                                v, v_, position[0], position_[0], dw, done)
            state = state_

            position = position_
            total_steps += 1
            print("epi：", epi)
            print("count=", replay_buffer.count)
            print("TTTTTTTTT")

            # 策略更新
            if replay_buffer.count == config.batch_size:
                # ================
                # 获取专家数据
                select_expert_data(expert_replay_buffer)
                # ================
                agent.update(replay_buffer, expert_replay_buffer, total_steps)

                replay_buffer.count = 0

            rewards_temp.append(r_vis)
            # if dw:
            #     r -= 50
            #     rewards_abs.append(abs(r))
            # else:
            #     rewards_abs.append(abs(r))

        rewards_vis.append(np.sum(rewards_temp) / target_reward)
        distance_temp = distance_caculation(posx, posy, posz)
        distance_vis.append(distance_temp / target_reward)

        if epi % config.tst_gap == 0:
            for ij in range(config.tst_pool):
                print('test-{0}-{1}'.format(epi, ij))
                tst(agent, env_test, config, epi, ij, rewards_tst_vis, distance_tst_vis)

        if epi % 200 == 0 and epi > 0:
            rewards_map_random(1, epi, rewards_vis, distance_vis)

        if epi % 500 == 0 and epi > 0:
            torch.save(agent.actor.state_dict(), "./figure/random/model_test_endR/modela_{}.pth".format(epi))
            torch.save(agent.fsn.state_dict(), "./figure/random/model_test_endR/fsna_{}.pth".format(epi))
            torch.save(agent.discriminator.state_dict(), "./figure/random/model_test_endR/disca_{}.pth".format(epi))


        # torch.save(agent.actor.state_dict(), "./model_test_endR/modela{}_.pth".format(episodes))
        # if cur_pos == ins_start and \
        #         end == end_ori and \
        #         target_reward == rewards[-1]:
        #     print("FIND.............................")
        #     rewards_map_random(1, epi, rewards_vis)
    agent.writer.close()

if __name__ == "__main__":
    train()




















