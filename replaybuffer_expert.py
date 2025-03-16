import torch
import numpy as np

from config import config

# torch.manual_seed(config.seed_number)
# np.random.seed(config.seed_number)

class ReplayBufferExpert:
    def __init__(self, args):
        self.s = np.zeros((args.batch_size, 18, args.img_size, args.img_size))
        self.a = np.zeros((args.batch_size, 1))
        self.s_ = np.zeros((args.batch_size, 18, args.img_size, args.img_size))
        self.pos = np.zeros((args.batch_size, 16))
        self.pos_ = np.zeros((args.batch_size, 16))
        self.act = np.zeros((args.batch_size, 10))
        # self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))  # is_terminal
        self.count = 0

    def store(self, s, s_, pos, pos_, act, done):
        self.s[self.count] = s
        self.s_[self.count] = s_
        self.pos[self.count] = pos
        self.pos_[self.count] = pos_
        self.act[self.count] = act
        # self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1


    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        pos = torch.tensor(self.pos, dtype=torch.float)
        pos_ = torch.tensor(self.pos_, dtype=torch.float)
        act = torch.tensor(self.act, dtype=torch.float)
        # dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, s_, pos, pos_, act, done
