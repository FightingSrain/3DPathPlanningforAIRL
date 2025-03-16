import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
from config import config

# torch.manual_seed(config.seed_number)
from torch.distributions import Beta, Normal
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# class TReLU(nn.Module):
#     def __init__(self):
#             super(TReLU, self).__init__()
#             self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
#             self.alpha.data.fill_(0)
#
#     def forward(self, x):
#         x = F.relu(x - self.alpha) + self.alpha
#         return x


class FSN(nn.Module):
    def __init__(self,):
        super(FSN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(8),
            nn.ELU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            nn.ELU(),
        )

        self.maxpool = torch.nn.AdaptiveMaxPool2d(1)
        # self.maxpool = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=False)
        kernel = torch.ones((1, 32, 3, 3))
        # kernel[:, :, 16, 16] = 1
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.bias = nn.Parameter(data=torch.zeros(1), requires_grad=False)


        self.pos = nn.Sequential(
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.ELU(),
            nn.Linear(16, 16),
            nn.LayerNorm(16),
            nn.ELU(),
        )

        self.share = nn.Sequential(
            nn.Linear(32 * 9, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
        )

        self.con_share = nn.Sequential(
            nn.Linear(128 + 16, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ELU(),
        )

    def pool(self, x):
        x = F.conv2d(x, self.weight, self.bias, stride=1, padding=0)
        return x

    def forward(self, s, p):
        b, _, _, _ = s.size()
        all_f = []
        for i in range(0, 9):
            temp = torch.Tensor(s[:, i: i + 2, :, :]).cuda()
            maxp = self.maxpool(self.conv(temp))
            # maxp = self.pool(self.conv(temp))
            all_f.append(maxp.view(b, -1))
        alls = torch.cat(all_f, 1).view(b, -1)
        sharef = self.share(alls)
        alls = torch.cat([sharef, self.pos(p.cuda())], 1)
        fea = self.con_share(alls)
        return fea

# fsn = FSN().to(device)
# ins = torch.ones((20, 18, 11, 11))
# p = torch.ones((20, 6))
# res = fsn(ins, p)
# print(res.size())