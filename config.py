import time

class config:
    # map parameters
    boundary = 5
    vis_map_size = (20, 20, 20)  # 活动空间的大小
    map_size = (vis_map_size[0] + boundary * 2,
                vis_map_size[1] + boundary * 2,
                vis_map_size[2] + boundary * 2)
    img_size = 1 + boundary * 2  # 小于活动空间的边长
    # =================
    obs_num = 45
    obs_size_min = (2, 2, 2)
    obs_size_max = (4, 4, 4)
    # =================
    is_random = True

    # train parameters
    batch_size = 8192 # 8192
    mini_batch_size = 128  # 128
    max_train_steps = int(3e8)
    sample_steps = 512  # 500
    is_use_sample_steps = True

    # agent parameters
    lr_a = 1e-4
    lr_d = 1e-4
    lr_fsn = 3e-4
    gamma = 0.99
    lamda = 0.98
    epsilon = 0.25
    K_epochs = 1
    entropy_coef = 0.03
    is_target = False
    reward_scale = 10

    # net parameters
    state_dim = 128
    action_dim = 65

    # tst parameters
    tst_gap = 50  # 50

    tst_pool = 5

    # seed
    seed_number = 1111
