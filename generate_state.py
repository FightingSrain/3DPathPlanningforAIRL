import numpy as np
from config import config
# np.random.seed(config.seed_number)
from map_3D import generate_map, render_map
from config import config
import matplotlib.pyplot as plt


def generate_state(map, cur, target):
    # map[cur[0], cur[1], cur[2]] = 0
    # map[target[0], target[1], target[2]] = 0

    x_slice_n = np.concatenate([[map[cur[0], cur[1]-5:cur[1]+6, cur[2]-5:cur[2]+6]],
                                [map[target[0], target[1]-5:target[1]+6, target[2]-5:target[2]+6]]], 0)
    # x_slice_n_1 = np.concatenate([[map[cur[0] - 1, :, :]], [map[target[0], :, :]]], 0)
    # x_slice_n_2 = np.concatenate([[map[cur[0] - 2, :, :]], [map[target[0], :, :]]], 0)
    # x_slice_n1 = np.concatenate([[map[cur[0] + 1, :, :]], [map[target[0], :, :]]], 0)
    # x_slice_n2 = np.concatenate([[map[cur[0] + 2, :, :]], [map[target[0], :, :]]], 0)
    # plt.imshow(state[0, 0])
    # plt.show()
    # print(x_slice_n.shape)
    # print("GGGGG")

    deep_x_1 = np.concatenate(
        [[map[cur[0] - 1, cur[1]-5:cur[1]+6, cur[2]-5:cur[2]+6] * 0.66 +
          (1 - map[cur[0] - 1, cur[1]-5:cur[1]+6, cur[2]-5:cur[2]+6]) *
          map[cur[0] - 2, cur[1]-5:cur[1]+6, cur[2]-5:cur[2]+6] * 0.33],
                               [map[target[0], target[1]-5:target[1]+6, target[2]-5:target[2]+6]]], 0)
    deep_x1 = np.concatenate(
        [[map[cur[0] + 1, cur[1]-5:cur[1]+6, cur[2]-5:cur[2]+6] * 0.66 +
          (1 - map[cur[0] + 1, cur[1]-5:cur[1]+6, cur[2]-5:cur[2]+6]) *
          map[cur[0] + 2, cur[1]-5:cur[1]+6, cur[2]-5:cur[2]+6] * 0.33],
           [map[target[0], target[1]-5:target[1]+6, target[2]-5:target[2]+6]]], 0)
    deep_x_1 = np.where(deep_x_1 == -0.33, -0.5, deep_x_1)
    deep_x_1 = np.where(deep_x_1 == -0.165, -0.5, deep_x_1)
    deep_x1 = np.where(deep_x1 == -0.33, -0.5, deep_x1)
    deep_x1 = np.where(deep_x1 == -0.165, -0.5, deep_x1)
    # print(deep_x_1.shape)
    # print(deep_x1.shape)

    # -----------
    y_slice_n = np.concatenate([[map[cur[0]-5:cur[0]+6, cur[1], cur[2]-5:cur[2]+6]],
                                [map[target[0]-5:target[0]+6, target[1], target[2]-5:target[2]+6]]], 0)
    # y_slice_n_1 = np.concatenate([[map[:, cur[1] - 1, :]], [map[:, target[1], :]]], 0)
    # y_slice_n_2 = np.concatenate([[map[:, cur[1] - 2, :]], [map[:, target[1], :]]], 0)
    # y_slice_n1 = np.concatenate([[map[:, cur[1] + 1, :]], [map[:, target[1], :]]], 0)
    # y_slice_n2 = np.concatenate([[map[:, cur[1] + 2, :]], [map[:, target[1], :]]], 0)
    # print(y_slice_n.shape)
    # plt.imshow(y_slice_n[1])
    # plt.show()
    deep_y_1 = np.concatenate(
    [[map[cur[0]-5:cur[0]+6, cur[1] - 1, cur[2]-5:cur[2]+6] * 0.66 +
      (1 - map[cur[0]-5:cur[0]+6, cur[1] - 1, cur[2]-5:cur[2]+6]) *
      map[cur[0]-5:cur[0]+6, cur[1] - 2, cur[2]-5:cur[2]+6] * 0.33],
     [map[target[0]-5:target[0]+6,
          target[1],
          target[2]-5:target[2]+6]]], 0)
    deep_y1 = np.concatenate(
    [[map[cur[0]-5:cur[0]+6, cur[1] + 1, cur[2]-5:cur[2]+6] * 0.66 +
      (1 - map[cur[0]-5:cur[0]+6, cur[1] + 1, cur[2]-5:cur[2]+6]) *
      map[cur[0]-5:cur[0]+6, cur[1] + 2, cur[2]-5:cur[2]+6] * 0.33],
     [map[target[0]-5:target[0]+6, target[1], target[2]-5:target[2]+6]]], 0)
    deep_y_1 = np.where(deep_y_1 == -0.33, -0.5, deep_y_1)
    deep_y_1 = np.where(deep_y_1 == -0.165, -0.5, deep_y_1)
    deep_y1 = np.where(deep_y1 == -0.33, -0.5, deep_y1)
    deep_y1 = np.where(deep_y1 == -0.165, -0.5, deep_y1)

    # -----------
    z_slice_n = np.concatenate([[map[cur[0]-5:cur[0]+6, cur[1]-5:cur[1]+6, cur[2]]],
                                [map[target[0]-5:target[0]+6, target[1]-5:target[1]+6, target[2]]]], 0)
    # z_slice_n_1 = np.concatenate([[map[:, :, cur[2] - 1]], [map[:, :, target[2]]]], 0)
    # z_slice_n_2 = np.concatenate([[map[:, :, cur[2] - 2]], [map[:, :, target[2]]]], 0)
    # z_slice_n1 = np.concatenate([[map[:, :, cur[2] + 1]], [map[:, :, target[2]]]], 0)
    # z_slice_n2 = np.concatenate([[map[:, :, cur[2] + 2]], [map[:, :, target[2]]]], 0)
    deep_z_1 = np.concatenate(
    [[map[cur[0]-5:cur[0]+6, cur[1]-5:cur[1]+6, cur[2] - 1] * 0.66 +
      (1 - map[cur[0]-5:cur[0]+6, cur[1]-5:cur[1]+6, cur[2] - 1]) *
      map[cur[0]-5:cur[0]+6, cur[1]-5:cur[1]+6, cur[2] - 2] * 0.33],
     [map[target[0]-5:target[0]+6, target[1]-5:target[1]+6, target[2]]]], 0)
    deep_z1 = np.concatenate(
    [[map[cur[0]-5:cur[0]+6, cur[1]-5:cur[1]+6, cur[2] + 1] * 0.66 +
      (1 - map[cur[0]-5:cur[0]+6, cur[1]-5:cur[1]+6, cur[2] + 1]) *
      map[cur[0]-5:cur[0]+6, cur[1]-5:cur[1]+6, cur[2] + 2] * 0.33],
     [map[target[0]-5:target[0]+6, target[1]-5:target[1]+6, target[2]]]], 0)
    deep_z_1 = np.where(deep_z_1 == -0.33, -0.5, deep_z_1)
    deep_z_1 = np.where(deep_z_1 == -0.165, -0.5, deep_z_1)
    deep_z1 = np.where(deep_z1 == -0.33, -0.5, deep_z1)
    deep_z1 = np.where(deep_z1 == -0.165, -0.5, deep_z1)
    # -----------
    x_f = np.concatenate([[deep_x_1], [x_slice_n], [deep_x1]], 1)
    # print(x_f.shape)
    y_f = np.concatenate([[deep_y_1], [y_slice_n], [deep_y1]], 1)
    z_f = np.concatenate([[deep_z_1], [z_slice_n], [deep_z1]], 1)

    state = np.concatenate([x_f, y_f, z_f], 1)
    # print(state.shape)


    return state
# map, start, end = generate_map(config.vis_map_size,
#                                        obs_num=config.obs_num,
#                                        obs_size_min=config.obs_size_min,
#                                        obs_size_max=config.obs_size_max,
#                                        is_random=config.is_random)
# print(start)
# print(end)
# generate_state(map, start, end)