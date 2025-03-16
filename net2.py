import torch
import torch.nn.functional as F
import torch.nn as nn

from config import config

# torch.manual_seed(config.seed_number)
from torch.distributions import Beta, Normal
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self,):
        super(Actor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.maxpool = torch.nn.AdaptiveMaxPool2d(1)

        self.pos = nn.Sequential(
            nn.Linear(6, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )
        self.share = nn.Sequential(
            nn.Linear(144 * 9, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        self.policy1 = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 10),
            # nn.LayerNorm(10),
        )
        self.value = nn.Sequential(       # Critic
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            # nn.LayerNorm(1),
        )
        self.gru_cell = torch.nn.GRU(256+16, 256, 1)  # inputs size, hidden size, num layers

    def forward(self, s, p, h_in):
        all_m = []
        b, _, _, _ = s.size()
        for i in range(0, 9):
            temp = torch.Tensor(s[:, i: i+2, :, :]).cuda()
            maxp = self.conv(temp)
            all_m.append(maxp.view(b, -1))
        alls = torch.cat(all_m, 1)
        hid = alls.view(b, -1)
        ps = torch.cat([self.share(hid), self.pos(p.cuda())], 1).view(1, b, -1)
        g, h_out = self.gru_cell(ps, h_in.cuda())
        a_prob1 = torch.softmax(self.policy1(g.view(b, -1)), dim=1)
        value = self.value(g.view(b, -1))
        return a_prob1, value, h_out
