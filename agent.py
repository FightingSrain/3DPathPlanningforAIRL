import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
from config import config
from net import Actor
from fsn import FSN
from utils import init_net, soft_update, hard_update
from Discriminator import Discriminator
import os
import itertools
from torch.utils.tensorboard import SummaryWriter
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.manual_seed(config.seed_number)
# np.random.seed(config.seed_number)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPO_discrete:
    def __init__(self, config):
        self.batch_size = config.batch_size  # 4096
        self.mini_batch_size = config.mini_batch_size  # 128
        self.max_train_steps = config.max_train_steps  # int(8e5)
        self.lr_a = config.lr_a  # Learning rate of actor
        self.lr_d = config.lr_d  # Learning rate of actor
        self.lr_fsn = config.lr_fsn  # Learning rate of actor
        self.gamma = config.gamma  # Discount factor
        self.lamda = config.epsilon  # GAE parameter  广义优势估计器
        self.epsilon = config.epsilon  # PPO clip parameter
        self.K_epochs = config.K_epochs  # PPO parameter
        self.entropy_coef = config.entropy_coef  # Entropy coefficient
        self.tau = 0.001
        self.set_adam_eps = False
        self.use_grad_clip = False
        self.use_lr_decay = False
        self.use_adv_norm = True
        self.writer = SummaryWriter(log_dir='./figure/random/log/')
        self.update_step = 1

        self.actor = init_net(Actor().to(device), 'orthogonal', gpu_ids=[])
        self.fsn = init_net(FSN().to(device), 'orthogonal', gpu_ids=[])
        self.discriminator = init_net(Discriminator().to(device), 'orthogonal', gpu_ids=[])
        if config.is_target:
            self.target_actor = init_net(Actor().to(device), 'orthogonal', gpu_ids=[])
            self.target_discriminator = init_net(Discriminator().to(device), 'orthogonal', gpu_ids=[])
            hard_update(self.target_discriminator, self.discriminator)
            hard_update(self.target_actor, self.actor)

        # if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
        #     self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
        # else:
        self.optimizer_actor = torch.optim.Adam(itertools.chain(self.actor.parameters(), self.fsn.parameters()), lr=self.lr_a, betas=(0.5, 0.999))
        # self.optimizer_discr = torch.optim.Adam(itertools.chain(self.discriminator.parameters(), self.fsn.parameters()), lr=self.lr_d, betas=(0.5, 0.999))
        self.optimizer_discr = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_d, betas=(0.5, 0.999))

        # self.optimizer_fsn = torch.optim.Adam(self.fsn.parameters(), lr=self.lr_fsn)

    def evaluate(self, s):  # When evaluating the policy, we select the action with the highest probability
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        a_prob = self.actor(s).detach().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def choose_action(self, s, p, test=False, target=False):
        s = torch.tensor(s, dtype=torch.float)
        p = torch.tensor(p, dtype=torch.float)
        # h = torch.tensor(h, dtype=torch.float)
        with torch.no_grad():
            with torch.no_grad():
                if target:
                    if test:
                        aprob1, value = self.target_actor(s, p, self.fsn.eval())
                    else:
                        aprob1, value = self.target_actor(s, p, self.fsn)
                else:
                    if test:
                        aprob1, value = self.actor(s, p, self.fsn.eval())
                    else:
                        aprob1, value = self.actor(s, p, self.fsn)
            print(aprob1)
            print("HHHHHHHH")
            dist1 = Categorical(probs=aprob1)
            if test:
                a1 = torch.argmax(aprob1, 1)
            else:
                a1 = dist1.sample()
            a_logprob1 = dist1.log_prob(a1)
        return a1.cpu().numpy()[0], a_logprob1.cpu().numpy()[0], \
               value.cpu().numpy()[0]

    def update(self, replay_buffer, expert_replay_buffer, total_steps):
        s, a1, a_one_hot, a_one_hot_, a_logprob1, r, s_, v, v_, p, p_, dw, done = replay_buffer.numpy_to_tensor()
        real_s, real_s_, real_p, real_p_, act_oh, real_done = expert_replay_buffer.numpy_to_tensor()
        # print(p.size())
        # print(p_.size())
        # print("TTTTTTTTTTTTTT")
        """
            Calculate the advantage using GAE
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        """
        # 我们需要先更新Critic，并计算出TD-error。再用TD-error更新Actor。
        adv = []  # reward
        gae = 0  # Critic网络更新
        with torch.no_grad():  # adv and v_target have no gradient
            vs = v
            vs_ = v_
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs  # td_error
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            # view() : Returns a new tensor with the same data as the self tensor but of a different shape.
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        for _ in range(self.K_epochs):
            flg = 0
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                with torch.no_grad():
                    log_pis_exp = self.actor.eval_log_pi(
                        real_s[index], real_p[index], self.fsn)
                self.update_gan(s[index], p[index], s_[index], p_[index], done[index], a_logprob1[index],
                                real_s[index], real_p[index], real_s_[index], real_p_[index], real_done[index],
                                log_pis_exp,
                                self.update_step)
                # if flg > 10:
                #     break
                flg += 1
        # Optimize policy for K epochs:         Actor网络更新
        for _ in range(self.K_epochs):
            flg = 0
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                # if flg > 10:
                #     break
                # flg += 1

                aprob_1, v_s = self.actor(s[index], p[index], self.fsn)

                dist_now1 = Categorical(probs=aprob_1)
                dist_entropy1 = dist_now1.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now1 = dist_now1.log_prob(a1[index].squeeze().cuda()).view(-1,
                                                                                     1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios1 = torch.exp(a_logprob_now1 - a_logprob1[index].cuda())  # shape(mini_batch_size X 1)
                # ratio调整了每个采样更新的步长。它越小，表示信任域越窄，策略更新越谨慎，从避免让新旧策略差异过大

                surr1_1 = ratios1 * adv[index].cuda()  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2_1 = torch.clamp(ratios1, 1 - self.epsilon, 1 + self.epsilon) * adv[index].cuda()  # 对估计优势的函数进行裁剪
                actor_loss1 = -torch.min(surr1_1,
                                         surr2_1) - self.entropy_coef * dist_entropy1  # shape(mini_batch_size X 1)
                # -------------------------------------
                critic_loss = F.mse_loss(v_target[index].cuda(), v_s)

                # critic_loss = F.smooth_l1_loss(v_target[index].cuda(), v_s)
                # self.optimizer_fsn.zero_grad()
                self.optimizer_actor.zero_grad()
                (actor_loss1.mean() + critic_loss).backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()
                # with torch.no_grad():
                #     log_pis_exp = self.actor.eval_log_pi(
                #         real_s[index], real_p[index])
                #     # a_probs, _ = self.actor(real_s[index], real_p[index], self.fsn)
                #     # dists = Categorical(probs=a_probs)
                #     # a_s = dists.sample()
                #     # logprob = dists.log_prob(a_s)
                #
                # self.update_gan(s[index], p[index], s_[index], p_[index], done[index], a_logprob1[index],
                #                 real_s[index], real_p[index], real_s_[index], real_p_[index], real_done[index], log_pis_exp,
                #                 self.update_step)

                if config.is_target:
                    soft_update(self.target_actor, self.actor, self.tau)
                # self.optimizer_fsn.step()

                self.writer.add_scalar(tag='Loss/entropy_loss', scalar_value=dist_entropy1.mean(), global_step=self.update_step)
                self.writer.add_scalar(tag='Loss/value_loss', scalar_value=critic_loss.mean(), global_step=self.update_step)
                self.writer.add_scalar(tag='Loss/policy_loss', scalar_value=actor_loss1.mean(), global_step=self.update_step)
                self.update_step += 1


        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def update_gan(self, s, p, s_, p_, done, logpi, r_s, r_p, r_s_, r_p_, r_done, r_logpi, total_steps):
        s = s.detach()
        p = p.detach()
        s_ = s_.detach()
        p_ = p_.detach()
        done = done.detach()
        logpi = logpi.detach()
        r_s = r_s.detach()
        r_p = r_p.detach()
        r_s_ = r_s_.detach()
        r_p_ = r_p_.detach()
        r_done = r_done.detach()
        r_logpi = r_logpi.detach()


        D_real = self.discriminator(r_s, r_p, r_s_, r_p_, r_logpi, self.gamma, r_done.cuda(), self.fsn)
        D_fake = self.discriminator(s, p, s_, p_, logpi, self.gamma, done.cuda(), self.fsn)
        # D_cost = ((D_fake) ** 2).mean() + ((D_real - 1) ** 2).mean()
        D_cost = -F.logsigmoid(D_real).mean() - F.logsigmoid(-D_fake).mean()
        print("Cost:", D_cost.data)
        print("Dfake:", D_fake.mean().data)
        print("Dreal:", D_real.mean().data)
        self.optimizer_discr.zero_grad()
        (D_cost).backward()
        self.optimizer_discr.step()
        self.writer.add_scalar(tag='Loss/D_loss', scalar_value=D_cost, global_step=total_steps)
        # Discriminator's accuracies.
        with torch.no_grad():
            acc_pi = (D_fake < 0).float().mean().item()
            acc_exp = (D_real > 0).float().mean().item()
        self.writer.add_scalar('stats/acc_pi', acc_pi, total_steps)
        self.writer.add_scalar('stats/acc_exp', acc_exp, total_steps)
        if config.is_target:
            soft_update(self.target_discriminator, self.discriminator, self.tau)
    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_d * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_discr.param_groups:
            p['lr'] = lr_c_now
        # ==========================

