import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.utils.spectral_norm as spectral_norm
from config import config

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
class Discriminator(nn.Module):
    def __init__(self,):
        super(Discriminator, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(2),
        #     nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        # )
        # self.maxpool = torch.nn.AdaptiveMaxPool2d(1)
        #
        # self.pos = nn.Sequential(
        #     (nn.Linear(6, 16)),
        #     nn.LayerNorm(16),
        #     nn.ReLU(),
        #     (nn.Linear(16, 16)),
        #     nn.LayerNorm(16),
        #     nn.ReLU(),
        # )
        # self.share = nn.Sequential(
        #     (nn.Linear(32 * 9, 128)),
        #     nn.LayerNorm(128),
        #     nn.ReLU(),
        #     (nn.Linear(128, 128)),
        #     nn.LayerNorm(128),
        #     nn.ReLU()
        # )
        self.g = nn.Sequential(  # Critic
            (nn.Linear(128, 128)),
            nn.LayerNorm(128),
            nn.ELU(),
            (nn.Linear(128, 128)),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
        self.h = nn.Sequential(  # Critic
            (nn.Linear(128, 128)),
            nn.LayerNorm(128),
            nn.ELU(),
            (nn.Linear(128, 128)),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 1),
        )
    def f(self, s, p, s_, p_, gamma, dones, fsn):
        b, _, _, _ = s.size()
        ps = fsn(s, p)  # .detach()
        # ------------------------------
        ps_ = fsn(s_, p_)  # .detach()
        rs = self.g(ps)
        vs = self.h(ps)
        next_vs = self.h(ps_)
        return rs + gamma * (1 - dones) * next_vs - vs

    def forward(self, s, p, s_, p_, log_pi, gamma, dones, fsn):
        return self.f(s, p, s_, p_, gamma, dones, fsn) - log_pi.cuda()

    def cal_reward(self, s, p, s_, p_, log_pi, gamma, dones, fsn):
        with torch.no_grad():
            logits = self.forward(s, p, s_, p_, log_pi, gamma, dones, fsn)
            return -F.logsigmoid(-logits)



