import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
from config import config
from torch.distributions import Categorical
# torch.manual_seed(config.seed_number)
from torch.distributions import Beta, Normal
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TReLU(nn.Module):
    def __init__(self):
            super(TReLU, self).__init__()
            self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x

class Actor(nn.Module):
    def __init__(self,):
        super(Actor, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(2, 32, kernel_size=5, stride=2, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        # )
        # self.maxpool = torch.nn.AdaptiveMaxPool2d(1)
        #
        # self.pos = nn.Sequential(
        #     nn.Linear(6, 16),
        #     nn.LayerNorm(16),
        #     nn.ReLU(),
        #     nn.Linear(16, 16),
        #     nn.LayerNorm(16),
        #     nn.ReLU(),
        # )
        # self.share = nn.Sequential(
        #     nn.Linear(16 * 9, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64-16),
        #     nn.LayerNorm(64-16),
        #     nn.ReLU()
        # )

        # self.policy = nn.Sequential(
        #     nn.Linear(128, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 10),
        # )
        #
        # self.value = nn.Sequential(       # Critic
        #     nn.Linear(128, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1),
        # )

        self.policy = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LayerNorm(10),
        )

        self.value = nn.Sequential(       # Critic
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.LayerNorm(1),
        )

        # self.gru_cell = torch.nn.GRU(64, 64, 1)  # inputs size, hidden size, num layers

    def forward(self, s, p, fsn):
        b, _, _, _ = s.size()
        g = fsn(s, p)
        a_prob = torch.softmax(self.policy(g.view(b, -1)), dim=1)
        value = self.value(g.view(b, -1))
        return a_prob, value

    def eval_log_pi(self, s, p, fsn):
        a_probs , _ = self.forward(s, p, fsn)
        dists = Categorical(probs=a_probs)
        a_s = dists.sample()
        logprob = dists.log_prob(a_s)
        return logprob
