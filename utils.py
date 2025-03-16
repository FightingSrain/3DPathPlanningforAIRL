import pickle

import cv2
import numpy as np
import torch
from torch.nn import init
import matplotlib.pyplot as plt

from config import config

# torch.manual_seed(config.seed_number)
# np.random.seed(config.seed_number)

def hard_update(target, source):
    for m1, m2 in zip(target.modules(), source.modules()):
        m1._buffers = m2._buffers.copy()
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def to_one_hot(num):
    onehot = [0]*10
    onehot[num] = 1
    return np.asarray(onehot)


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def reward_caculation(start, end):  # , target=False
    reward = np.sqrt(np.square(start[0] - end[0]) + \
                    np.square(start[1] - end[1]) + \
                    np.square(start[2] - end[2]))
    reward *= config.reward_scale

    # if target:
    #     reward += 0  # 50

    # reward += 100
    return reward


def distance_caculation(posx, posy, posz):
    distance = []
    for i in range(len(posx)-1):
        distance.append(np.sqrt(np.square(posx[i+1] - posx[i]) + \
                    np.square(posy[i+1] - posy[i]) + \
                    np.square(posz[i+1] - posz[i])))
    distance_sum = np.sum(distance) * config.reward_scale
    return distance_sum


def vis_rgbimg(img, time=1):
    image = np.asanyarray(img[0, 0:3, :, :].transpose(1, 2, 0) * 255, dtype=np.uint8)
    image = np.squeeze(image)
    cv2.imshow("rerr", image)
    cv2.waitKey(time)


def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':  #正交
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='xavier', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net

def slip_average(reward, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(reward, window, 'same')
    return re


def rewards_map_random(y, epi, reward, distance):
    sigAv = slip_average(reward, 50)  # 做一下简单的滑动平均
    sigAv_dis = slip_average(distance, 50)

    x = range(1, epi+1)
    # plt.axhline(y=y, c='r', ls='--', lw=2)
    # plt.axhline(y=0.5, c='r', ls='--', lw=1)
    # plt.axhline(y=0.6, c='r', ls='--', lw=1)
    # plt.axhline(y=0.7, c='r', ls='--', lw=1)
    # plt.axhline(y=0.8, c='r', ls='--', lw=1)
    # plt.axhline(y=0.9, c='r', ls='--', lw=1)

    #plt.title('train-js50+relu+max+layernorm+gam0.99')
    #plt.grid()

    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    ax[0].set_ylim(-1, 1.1)
    ax[0].axhline(y=y, c='r', ls='--', lw=2)
    ax[0].axhline(y=0.5, c='r', ls='--', lw=1)
    ax[0].axhline(y=0.6, c='r', ls='--', lw=1)
    ax[0].axhline(y=0.7, c='r', ls='--', lw=1)
    ax[0].axhline(y=0.8, c='r', ls='--', lw=1)
    ax[0].axhline(y=0.9, c='r', ls='--', lw=1)
    ax[0].set_title('train-js50+relu+max+layernorm+gam0.99')
    ax[0].plot(x, reward, color='green', linestyle='-', linewidth=5, label='rewards', alpha=0.2)
    ax[0].plot(x, sigAv, color='blue', linestyle='-', linewidth=2, label='silp', alpha=0.8)
    ax[0].grid()

    ax[1].set_title('distance')
    ax[1].set_ylim(0, 50)
    ax[1].axhline(y=y, c='r', ls='--', lw=2)
    # ax[1].axhline(y=0.7, c='r', ls='--', lw=1)
    # ax[1].axhline(y=0.8, c='r', ls='--', lw=1)
    # ax[1].axhline(y=0.9, c='r', ls='--', lw=1)
    # ax[1].axhline(y=1.1, c='r', ls='--', lw=1)
    # ax[1].axhline(y=1.2, c='r', ls='--', lw=1)
    # ax[1].axhline(y=1.3, c='r', ls='--', lw=1)
    ax[1].plot(x, distance, color='green', linestyle='-', linewidth=5, label='rewards', alpha=0.2)
    ax[1].plot(x, sigAv_dis, color='blue', linestyle='-', linewidth=2, label='silp', alpha=0.8)
    ax[1].grid()
    #plt.plot(x,avg_reward, color='red', linestyle='-', linewidth=1,)
    if y == reward[-1]:
        plt.savefig('./figure/random/' + 'final-rewards-js50+relu+max+layernorm-gam0.99-' + str(epi) +'.png')
    else:
        plt.savefig('./figure/random/' + 'rewards-js50+relu+max+layernorm-gam0.99-' + str(epi) + '.png')

    plt.pause(1)
    plt.close()


def rewards_map_random_tst(y, epi, reward, distance):
    #sigAv = slip_average(reward, 5)  # 做一下简单的滑动平均
    sigAv = slip_average(reward, 50)  # 做一下简单的滑动平均
    sigAv_dis = slip_average(distance, 50)

    x = range(1, int(config.tst_pool * (epi/config.tst_gap))+1)

    fig, ax = plt.subplots(1, 2, figsize=(20, 7))
    ax[0].set_ylim(-1, 1.1)
    ax[0].axhline(y=y, c='r', ls='--', lw=2)
    ax[0].axhline(y=0.5, c='r', ls='--', lw=1)
    ax[0].axhline(y=0.6, c='r', ls='--', lw=1)
    ax[0].axhline(y=0.7, c='r', ls='--', lw=1)
    ax[0].axhline(y=0.8, c='r', ls='--', lw=1)
    ax[0].axhline(y=0.9, c='r', ls='--', lw=1)
    ax[0].set_title('tst-js50+relu+max+layernorm+gam0.99')
    ax[0].plot(x, reward, color='green', linestyle='-', linewidth=5, label='rewards', alpha=0.2)
    ax[0].plot(x, sigAv, color='blue', linestyle='-', linewidth=2, label='silp', alpha=0.8)
    ax[0].grid()


    ax[1].set_title('distance')
    ax[1].set_ylim(0, 50)
    ax[1].axhline(y=y, c='r', ls='--', lw=2)
    # ax[1].axhline(y=0.7, c='r', ls='--', lw=1)
    # ax[1].axhline(y=0.8, c='r', ls='--', lw=1)
    # ax[1].axhline(y=0.9, c='r', ls='--', lw=1)
    # ax[1].axhline(y=1.1, c='r', ls='--', lw=1)
    # ax[1].axhline(y=1.2, c='r', ls='--', lw=1)
    # ax[1].axhline(y=1.3, c='r', ls='--', lw=1)
    ax[1].plot(x, distance, color='green', linestyle='-', linewidth=5, label='rewards', alpha=0.2)
    ax[1].plot(x, sigAv_dis, color='blue', linestyle='-', linewidth=2, label='silp', alpha=0.8)
    ax[1].grid()

    # plt.title('tst-js50+relu+max+layernorm+gam0.99')
    # plt.grid()
    # plt.plot(x, reward, color='green', linestyle='-', linewidth=1, label='rewards', alpha=0.5)

    #plt.plot(x,avg_reward, color='red', linestyle='-', linewidth=1,)
    if y == reward[-1]:
        plt.savefig('./figure/random/tst/' + 'f-tst-js50+relu+max+layernorm-gam0.99-' + str(epi) +'.png')
    else:
        plt.savefig('./figure/random/tst/' + 'tst-js50+relu+max+layernorm-gam0.99-' + str(epi) + '.png')
    #plt.show()

    plt.pause(1)
    plt.close()


def record_inf(is_which, epi, ij, map, posx, posy, posz, start, end):
    if is_which == 1:
        map_path = "./figure/random/information/reward_succ/{0}-{1}-map.pkl".format(str(epi), str(ij))
        other_path = "./figure/random/information/reward_succ/{0}-{1}-other.pkl".format(str(epi), str(ij))
    elif is_which == 2:
        map_path = "./figure/random/information/dist_succ/{0}-{1}-map.pkl".format(str(epi), str(ij))
        other_path = "./figure/random/information/dist_succ/{0}-{1}-other.pkl".format(str(epi), str(ij))
    elif is_which == 3:
        map_path = "./figure/random/information/fail/{0}-{1}-map.pkl".format(str(epi), str(ij))
        other_path = "./figure/random/information/fail/{0}-{1}-other.pkl".format(str(epi), str(ij))
    elif is_which == 4:
        map_path = "./figure/random/information/train/{0}-{1}-map.pkl".format(str(epi), str(ij))
        other_path = "./figure/random/information/train/{0}-{1}-other.pkl".format(str(epi), str(ij))

    with open(map_path, "wb") as f:
        pickle.dump(map, f)

    other = [posx] + [posy] + [posz] + [start] + [end]
    # print(other)
    other = np.array(other, dtype=object)
    # print(other.shape)

    with open(other_path, "wb") as f:
        pickle.dump(other, f)

