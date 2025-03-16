import torch
import numpy as np

from config import config

# torch.manual_seed(config.seed_number)
# np.random.seed(config.seed_number)

class ReplayBuffer:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, 18, args.img_size, args.img_size))
        # self.real_s = np.zeros((args.batch_size, 18, args.img_size, args.img_size))

        self.a1 = np.zeros((args.batch_size, 1))
        self.a_one_hot = np.zeros((args.batch_size, 10))
        self.a_one_hot_ = np.zeros((args.batch_size, 10))
        self.a_logprob1 = np.zeros((args.batch_size, 1))

        # self.a2 = np.zeros((args.batch_size, 1))
        # self.a_logprob2 = np.zeros((args.batch_size, 1))

        self.r = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, 18, args.img_size, args.img_size))
        self.v = np.zeros((args.batch_size, 1))
        self.v_ = np.zeros((args.batch_size, 1))
        # self.h = np.zeros((1, args.batch_size, 128))
        self.pos = np.zeros((args.batch_size, 16))
        self.pos_ = np.zeros((args.batch_size, 16))
        # self.real_pos = np.zeros((args.batch_size, 6))
        # self.start = np.zeros((args.batch_size, 3))
        # self.end = np.zeros((args.batch_size, 3))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))  # is_terminal
        self.count = 0

    def store(self, s, a1, a_one_hot, a_one_hot_, a_logprob1, r, s_, v, v_, pos, pos_, dw, done):
        self.s[self.count] = s
        # self.real_s[self.count] = real_s
        self.a1[self.count] = a1
        self.a_one_hot[self.count] = a_one_hot
        self.a_one_hot_[self.count] = a_one_hot_
        self.a_logprob1[self.count] = a_logprob1
        # self.a2[self.count] = a2
        # self.a_logprob2[self.count] = a_logprob2
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.v[self.count] = v
        self.v_[self.count] = v_
        # self.h[:, self.count, :] = h
        self.pos[self.count] = pos
        self.pos_[self.count] = pos_
        # self.real_pos[self.count] = real_pos
        # self.start[self.count] = start
        # self.end[self.count] = end
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1


    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        # real_s = torch.tensor(self.real_s, dtype=torch.float)
        a1 = torch.tensor(self.a1, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_one_hot = torch.tensor(self.a_one_hot, dtype=torch.long)
        a_one_hot_ = torch.tensor(self.a_one_hot_, dtype=torch.long)
        a_logprob1 = torch.tensor(self.a_logprob1, dtype=torch.float)
        # a2 = torch.tensor(self.a2, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        # a_logprob2 = torch.tensor(self.a_logprob2, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        v = torch.tensor(self.v, dtype=torch.float)
        v_ = torch.tensor(self.v_, dtype=torch.float)
        # h = torch.tensor(self.h, dtype=torch.float)
        pos = torch.tensor(self.pos, dtype=torch.float)
        pos_ = torch.tensor(self.pos_, dtype=torch.float)
        # real_pos = torch.tensor(self.real_pos, dtype=torch.float)
        # start = torch.tensor(self.start, dtype=torch.float)
        # end = torch.tensor(self.end, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a1, a_one_hot, a_one_hot_, a_logprob1, r, s_, v, v_, pos, pos_, dw, done
