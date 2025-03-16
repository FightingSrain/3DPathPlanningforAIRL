import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

from config import config

# np.random.seed(config.seed_number)

import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D


def generate_map(real_sizes, obs_num, obs_size_min, obs_size_max, boundary=5, is_random=True, fix_start=(2, 2, 2),
                 fix_end=(12, 12, 12)):
    """
    :param real_sizes: 可见地图大小 例：（14，14，14）或（24，24，24）
    :param obs_num: 障碍物数量 建议 25到40， 和地图大小有关
    :param obs_size: 障碍物尺寸 例：（3，3，3）或（5，5，5）
    :param is_random: 是否随机起始，终止点， 随机：True，固定：False
    :param fix_start: 固定起始点坐标 例：（2，2，2），（3，3，3）
    :param fix_end: 固定终止点坐标 例：（11，11，11），（21，21，21）
    :return: 地图矩阵map，起始点坐标列表start，终止点坐标列表end
    """

    size = (real_sizes[0] + boundary * 2,
            real_sizes[1] + boundary * 2,
            real_sizes[2] + boundary * 2)
    map = np.zeros(size)  # 扩展后的地图
    seed = []  # 障碍物起始点
    for i in range(obs_num):  # 障碍物个数
        x = np.random.randint(0, size[0] - 0)
        y = np.random.randint(0, size[1] - 0)
        z = np.random.randint(0, size[2] - 0)
        seed.append((x, y, z))

    # 随机障碍物大小, 随机 长，宽， 高
    for i in range(len(seed)):
        obs_x = np.random.randint(obs_size_min[0], obs_size_max[0] + 1)
        obs_y = np.random.randint(obs_size_min[1], obs_size_max[1] + 1)
        obs_z = np.random.randint(obs_size_min[2], obs_size_max[2] + 1)

        flag = np.random.randint(0, 6)
        if flag == 0:
            map[seed[i][0]:np.clip(seed[i][0] + obs_x, a_min=0, a_max=size[0]),
            seed[i][1]:np.clip(seed[i][1] + obs_y, a_min=0, a_max=size[1]),
            seed[i][2]:np.clip(seed[i][2] + obs_z, a_min=0, a_max=size[2])] = 1
        elif flag == 1:
            map[np.clip(seed[i][0] - obs_x, a_min=0, a_max=size[0]):seed[i][0],
            seed[i][1]:np.clip(seed[i][1] + obs_y, a_min=0, a_max=size[1]),
            seed[i][2]:np.clip(seed[i][2] + obs_z, a_min=0, a_max=size[2])] = 1
        elif flag == 2:
            map[seed[i][0]:np.clip(seed[i][0] + obs_x, a_min=0, a_max=size[0]),
            np.clip(seed[i][1] - obs_y, a_min=0, a_max=size[1]):seed[i][1],
            seed[i][2]:np.clip(seed[i][2] + obs_z, a_min=0, a_max=size[2])] = 1
        elif flag == 3:
            map[np.clip(seed[i][0] - obs_x, a_min=0, a_max=size[0]):seed[i][0],
            np.clip(seed[i][1] - obs_y, a_min=0, a_max=size[1]):seed[i][1],
            seed[i][2]:np.clip(seed[i][2] + obs_z, a_min=0, a_max=size[2])] = 1
        elif flag == 4:
            map[seed[i][0]:np.clip(seed[i][0] + obs_x, a_min=0, a_max=size[0]),
            seed[i][1]:np.clip(seed[i][1] + obs_y, a_min=0, a_max=size[1]),
            np.clip(seed[i][2] - obs_z, a_min=0, a_max=size[2]):seed[i][2]] = 1
        else:
            map[seed[i][0]:np.clip(seed[i][0] + obs_x, a_min=0, a_max=size[0]),
            np.clip(seed[i][1] - obs_y, a_min=0, a_max=size[1]):seed[i][1],
            np.clip(seed[i][2] - obs_z, a_min=0, a_max=size[2]):seed[i][2]] = 1

        # map[seed[i][0]:seed[i][0] + obs_x,
        #     seed[i][1]:seed[i][1] + obs_y,
        #     seed[i][2]:seed[i][2] + obs_z] = 1
    # ----------------------------------------------------
    # 随机起始，终止点
    # start = []
    # end = []
    if is_random:
        while True:
            sx = np.random.randint(boundary, size[0] - boundary)
            sy = np.random.randint(boundary, size[1] - boundary)
            sz = np.random.randint(boundary, size[2] - boundary)
            # 随机终止坐标点
            ex = np.random.randint(boundary, size[0] - boundary)
            ey = np.random.randint(boundary, size[1] - boundary)
            ez = np.random.randint(boundary, size[2] - boundary)
            if map[sx, sy, sz] == 0 and map[ex, ey, ez] == 0 and sx != ex and sy != ey and sz != ez:
                # 起始终止标记
                map[sx, sy, sz] = 0.5  # 当前位置
                map[ex, ey, ez] = -0.5  # 目标位置
                break

        # start.append(sx)
        # start.append(sy)
        # start.append(sz)
        # end.append(ex)
        # end.append(ey)
        # end.append(ez)

        start = [sx, sy, sz]
        end = [ex, ey, ez]
        return map, start, end
    # ----------------------------------------------------
    # 固定起始终止点
    else:
        # for i in range(3):
        #     start.append(fix_start[i])
        #     end.append(fix_end[i])
        # 起始终止标记
        map[fix_start[0], fix_start[1], fix_start[2]] = 0.5
        map[fix_end[0], fix_end[1], fix_end[2]] = -0.5
        return map, fix_start, fix_end


def render_map(map, posx, posy, posz, start, end):
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_axes(Axes3D(fig))
    # ax = Axes3D(fig)
    cmap = plt.get_cmap('Blues')  # 'Wistia'
    colors = [cmap(i) for i in np.linspace(1, 0, map.shape[2])]
    c = np.empty(map.shape, dtype=object)
    for i in range(map.shape[2]):
        for a in range(map.shape[0]):
            for b in range(map.shape[1]):
                if map[a, b, i] == 1.0:
                    c[a, b, i] = colors[i]
                else:
                    c[a, b, i] = None

    sxyz = [start[0] + 0.5, start[1] + 0.5, start[2] + 0.5]
    exyz = [end[0] + 0.5, end[1] + 0.5, end[2] + 0.5]
    # 地图起始点，终止点清零
    map[start[0], start[1], start[2]] = 0
    map[end[0], end[1], end[2]] = 0

    ax.plot(xs=posx, ys=posy, zs=posz, linewidth=2, c='black')
    # ax.scatter(xs=posx, ys=posy, zs=posz, linewidth=1, c='red')
    ax.voxels(map[config.boundary:config.map_size[0] - config.boundary,
              config.boundary:config.map_size[1] - config.boundary,
              config.boundary:config.map_size[2] - config.boundary],

              facecolors=c[config.boundary:config.map_size[0] - config.boundary,
                         config.boundary:config.map_size[1] - config.boundary,
                         config.boundary:config.map_size[2] - config.boundary], shade=False)
    # ax.voxels(map, facecolors=c, shade=False)

    ax.scatter(xs=sxyz[0] - config.boundary,
               ys=sxyz[1] - config.boundary,
               zs=sxyz[2] - config.boundary, c='red', s=30, alpha=1, marker='^')
    ax.scatter(xs=exyz[0] - config.boundary,
               ys=exyz[1] - config.boundary,
               zs=exyz[2] - config.boundary, c='blue', s=30, alpha=1, marker='*')
    plt.title('deep pic')

    # plt.show()
    plt.pause(3)
    plt.close()


def record_inf(is_which, epi, ij, map, posx, posy, posz, start, end):
    if is_which == 1:
        # 如果没有文件夹，创建文件夹
        if not os.path.exists("./figure/random/information/reward_succ"):
            os.makedirs("./figure/random/information/reward_succ")
        map_path = "./figure/random/information/reward_succ/{0}-{1}-map.pkl".format(str(epi), str(ij))
        other_path = "./figure/random/information/reward_succ/{0}-{1}-other.pkl".format(str(epi), str(ij))
    elif is_which == 2:
        # 如果没有文件夹，创建文件夹
        if not os.path.exists("./figure/random/information/dist_succ"):
            os.makedirs("./figure/random/information/dist_succ")
        map_path = "./figure/random/information/dist_succ/{0}-{1}-map.pkl".format(str(epi), str(ij))
        other_path = "./figure/random/information/dist_succ/{0}-{1}-other.pkl".format(str(epi), str(ij))
    elif is_which == 3:
        # 如果没有文件夹，创建文件夹
        if not os.path.exists("./figure/random/information/fail"):
            os.makedirs("./figure/random/information/fail")
        map_path = "./figure/random/information/fail/{0}-{1}-map.pkl".format(str(epi), str(ij))
        other_path = "./figure/random/information/fail/{0}-{1}-other.pkl".format(str(epi), str(ij))
    elif is_which == 4:
        # 如果没有文件夹，创建文件夹
        if not os.path.exists("./figure/random/information/train"):
            os.makedirs("./figure/random/information/train")
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


def print_inf(epi, ij):
    ###Extract from file
    with open("./figure/random/information/{0}-{1}-map.pkl".format(str(epi), str(ij)), "rb") as f:
        map = pickle.load(f)
        # print(type(map))
        # print(map)

    with open("./figure/random/information/{0}-{1}-other.pkl".format(str(epi), str(ij)), "rb") as f:
        other = pickle.load(f)
        # print(type(other))
        # print(other)

    return map, other[0], other[1], other[2], other[3], other[4]

# # ---test---
# map_size = (14, 14, 14)
# obs_num = 5
# obs_size = (3, 3, 3)
# fix_start = (3, 3, 3)
# fix_end = (11, 11, 11)
# posx = [3.5, 4.5, 5.5]
# posy = [3.5, 4.5, 5.5]
# posz = [3.5, 4.5, 5.5]
#
# map, start, end = gen_map(map_size, obs_num, obs_size,
#                           is_random=True,
#                           fix_start=fix_start, fix_end=fix_end)
# render_map(map, posx, posy, posz, start, end)
