import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Categorical
from timm import create_model

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.hidden = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]


class ActorCritic(nn.Module):
    def __init__(self, feature_dim, state_dim, hidden_state_dim=1024, view_num = 1, fc_num = 1, policy_conv=True, same_gru=False, coarse='', batch_size=32):
        super(ActorCritic, self).__init__()
        
        # encoder with convolution layer for MobileNetV3, EfficientNet and RegNet
        if policy_conv:
            self.state_encoder = nn.Sequential(
                nn.Conv2d(feature_dim, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(int(state_dim * 32 / feature_dim), hidden_state_dim),
                nn.ReLU()
            )
        # encoder with linear layer for ResNet and DenseNet
        else:
            self.state_encoder = nn.Sequential(
                nn.BatchNorm1d(feature_dim),
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Linear(512, hidden_state_dim),
                nn.ReLU()
            )

        if same_gru and coarse:
            self.fc = Full_layer(feature_dim, hidden_state_dim, fc_num, class_num=4, batch_size=batch_size)
        elif same_gru:
            self.fc = Full_layer(feature_dim, hidden_state_dim, fc_num, batch_size=batch_size)
        else:
            self.fc = Full_layer(hidden_state_dim, hidden_state_dim, fc_num, batch_size=batch_size)
        
        self.actor = nn.ModuleList()
        if fc_num == 1:
            self.actor = nn.Sequential(
            nn.BatchNorm1d(hidden_state_dim),
            nn.Linear(hidden_state_dim, view_num),
            nn.Softmax(dim = 1))
        else:
            for _ in range(fc_num):
                self.actor.append(nn.Sequential(
                    nn.BatchNorm1d(hidden_state_dim),
                    nn.Linear(hidden_state_dim, view_num),
                    nn.Softmax(dim = 1)))

        self.critic = nn.Sequential(
            nn.BatchNorm1d(hidden_state_dim),
            nn.Linear(hidden_state_dim, 1))

        self.hidden_state_dim = hidden_state_dim
        self.policy_conv = policy_conv
        self.feature_dim = feature_dim
        self.feature_ratio = int(math.sqrt(state_dim/feature_dim))
        self.fc_num = fc_num

    def forward(self):
        raise NotImplementedError

    def act(self, state_ini, memory, args, step, restart_batch=False, training=False):
        if not self.policy_conv:
            state = state_ini.flatten(1)
        else:
            state = state_ini
        
        if not args.same_gru:
            state = self.state_encoder(state)

        state, _ = self.fc(state, step, restart=restart_batch)
        state = state[0]
        if self.fc_num == 1:
            action_output = self.actor(state)
        else:
            action_output = self.actor[step](state)
        batch_size = state.size(0)

        if training:
            action_dist = Categorical(probs = action_output)
            action = action_dist.sample()
            action_prob = torch.gather(action_output, dim = 1, index = action.view(-1, 1)).view(1, -1)
            action_logprob = torch.log(action_prob).cuda()

            memory.states.append(state_ini)
            memory.actions.append(action)
            memory.logprobs.append(torch.squeeze(action_logprob))

            return torch.squeeze(action.detach())
        else:
            _, top1_action = torch.max(action_output, 1)

            if args.overlap_mask:
                overlap_actions = torch.stack(memory.actions, 1)
                for i in range(batch_size):
                    action_output[i,:][overlap_actions[i,:]] = 0

            _, action = torch.max(action_output, 1)

            return torch.squeeze(action.detach()), torch.squeeze(top1_action.detach())

    def evaluate(self, state, action, args):
        seq_l = state.size(0)
        batch_size = state.size(1)
        if not self.policy_conv:
            state = state.flatten(2)

        if not args.same_gru:
            state_l = []
            for i in range(seq_l):
                state_l.append(self.state_encoder(state[i]).unsqueeze(0))
            state = torch.cat(state_l, 0)
        
        state_l = []
        for i in range(seq_l):
            state_output, cls_output = self.fc(state[i], i+1, restart=True)
            state_l.append(state_output)
        state = torch.cat(state_l, 0)

        action_output_l = []
        for i in range(seq_l):
            if self.fc_num == 1:
                action_output_l.append(self.actor[0](state[i]).unsqueeze(0))
            else:
                action_output_l.append(self.actor[i+1](state[i]).unsqueeze(0))
        action_output = torch.cat(action_output_l, 0)
        
        action_output = action_output.view(seq_l * batch_size, -1)
        state = state.view(seq_l * batch_size, -1)
        
        action_dist = Categorical(probs = action_output)
        action_prob = torch.gather(action_output, dim = 1, index = action.view(seq_l * batch_size, 1)).view(1, -1)
        action_logprobs = torch.log(action_prob).cuda()
        output_logprobs = torch.log(action_output).cuda()
        dist_entropy = action_dist.entropy()
        state_value = self.critic(state)

        return action_logprobs.view(seq_l, batch_size), \
               output_logprobs.view(seq_l, batch_size, args.view_num), \
               state_value.view(seq_l, batch_size), \
               dist_entropy.view(seq_l, batch_size), \
               cls_output

class PPO:
    def __init__(self, feature_dim, state_dim, hidden_state_dim, view_num, fc_num, policy_conv, same_gru, coarse, batch_size,
                 lr=0.0005, betas=(0.9, 0.999), gamma=0.7, K_epochs=1, eps_clip=0.2):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy = ActorCritic(feature_dim, state_dim, hidden_state_dim, view_num, fc_num, policy_conv, same_gru, coarse, batch_size).cuda()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        # self.optimizer = torch.optim.SGD([{'params': self.policy.parameters(), 'lr': lr}], 
        #     momentum=0.9, weight_decay=5e-4)

        self.policy_old = ActorCritic(feature_dim, state_dim, hidden_state_dim, view_num, fc_num, policy_conv, same_gru, coarse, batch_size).cuda()
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory, args, step, restart_batch=False, training=True):
        return self.policy_old.act(state, memory, args, step, restart_batch, training)

    def update(self, memory, target_var, criterion, args):
        if args.current_rewards:
            rewards = torch.cat(memory.rewards, 0).cuda()
        else:
            rewards = []
            discounted_reward = 0
            for reward in reversed(memory.rewards):
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.cat(rewards, 0).cuda()

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.stack(memory.states, 0).cuda().detach()
        old_actions = torch.stack(memory.actions, 0).cuda().detach()
        old_logprobs = torch.stack(memory.logprobs, 0).cuda().detach()

        if args.overlap_rewards:
            for i in range(rewards.size(1)):
                a = old_actions[:,i]
                overlap = [i for i,x in enumerate(a) if x in a[:i]]
                rewards[:,i][overlap] = -rewards.std()

        for _ in range(self.K_epochs):
            logprobs, _, state_values, dist_entropy, cls_output = self.policy.evaluate(old_states, old_actions, args)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            cls_loss = torch.zeros(cls_output.size(0), cls_output.size(1))
            if not args.coarse_checkpoint_path:
                for i in range(cls_output.size(0)):
                    cls_loss[i,:] = criterion(cls_output[i], target_var)
            loss = - torch.min(surr1, surr2) \
                   + 0.5 * self.MseLoss(state_values, rewards.view(state_values.size(0), state_values.size(1))) \
                   - 0.0001 * dist_entropy \
                   + args.ppo_fc * cls_loss.cuda() \
                   - args.perceptual * self.MseLoss(state_values[0:-1, :], state_values[1:, :])
  
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def multi_update(self, memory, target_var, criterion, args):
        # memory.rewards - step_num * batch_size * view_num
        rewards = torch.stack(memory.rewards, 0).cuda()
        old_states = torch.stack(memory.states, 0).cuda().detach()
        old_actions = torch.stack(memory.actions, 0).cuda().detach()
        old_logprobs = torch.stack(memory.logprobs, 0).cuda().detach()
        
        if args.overlap_rewards:
            # old_actions.size() = [6, 32], rewards.size() = [6, 32, 7]
            for i in range(rewards.size(0)):
                for j in range(rewards.size(1)):
                    rewards[i, j, : ][old_actions[:i, j]] = -rewards.std()

        view_num = rewards.size(2)
        step_num = rewards.size(0)
        
        # [step_num, batch_size, view_num] -> [step_num, batch_size * view_num]
        # step_num = 6, view_num = 7
        rewards = rewards.view(step_num, -1)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5) # torch.Size([6, 224])

        for _ in range(self.K_epochs):
            logprobs, multilogprobs, state_values, dist_entropy, cls_output = self.policy.evaluate(old_states, old_actions, args, step)
            ratios = torch.exp(multilogprobs - multilogprobs.detach())
            
            # [step_num, batch_size] -> [step_num, batch_size * view_num]
            state_values = state_values.unsqueeze(-1).repeat(1, 1, view_num).view(step_num, -1) # torch.Size([6, 224])
            dist_entropy = dist_entropy.unsqueeze(-1).repeat(1, 1, view_num).view(step_num, -1) # torch.Size([6, 224])
            ratios = ratios.view(step_num, -1) # torch.Size([6, 224])

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            cls_loss = torch.zeros(cls_output.size(0), cls_output.size(1))
            if not args.coarse_checkpoint_path:
                for i in range(cls_output.size(0)):
                    cls_loss[i,:] = criterion(cls_output[i], target_var)
            loss = - torch.min(surr1, surr2) \
                   + 0.5 * self.MseLoss(state_values, rewards) \
                   - 0.0001 * dist_entropy \
                   + args.ppo_fc * cls_loss.repeat(1, view_num).cuda() \
                   - args.perceptual * self.MseLoss(state_values[0:-1, :], state_values[1:, :])

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

class Full_layer(torch.nn.Module):
    def __init__(self, feature_num, hidden_state_dim=1024, fc_num = 1, fc_rnn='gru', class_num=1000, batch_size=32):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.feature_num = feature_num
        self.fc_num = fc_num

        self.hidden_state_dim = hidden_state_dim
        self.hidden = None
        self.fc_rnn = fc_rnn
        self.batch_size = batch_size
        self.fc = nn.ModuleList()

        # classifier with RNN for ResNet, DenseNet and RegNet
        if fc_rnn == 'gru':
            self.rnn = nn.GRU(self.feature_num, self.hidden_state_dim)
            if fc_num == 1:
                self.fc = nn.Linear(self.hidden_state_dim, self.class_num)
            else:
                for _ in range(self.fc_num):
                    self.fc.append(nn.Linear(self.hidden_state_dim, self.class_num))
        elif fc_rnn == 'lstm':
            self.rnn = nn.LSTM(self.feature_num, self.hidden_state_dim, 1)
            for _ in range(self.fc_num):
                self.fc.append(nn.Linear(self.hidden_state_dim, self.class_num))
        elif fc_rnn == 'transformer':
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_num, nhead=8)
            self.rnn = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
            for _ in range(self.fc_num):
                self.fc.append(nn.Linear(self.feature_num, self.class_num))
        # cascaded classifier for MobileNetV3 and EfficientNet
        elif fc_rnn == 'nfc':
            for i in range(self.fc_num):
                self.fc.append(nn.Linear(self.feature_num * (i+1), self.class_num))

    def forward(self, x, step, restart=False):
        if x.size(0) == self.batch_size:
            x = x.unsqueeze(0)

        if self.fc_rnn == 'gru':
            if restart:
                output, h_n = self.rnn(x, torch.zeros(1, x.size(1), self.hidden_state_dim).cuda())
                self.hidden = h_n
            else:
                output, h_n = self.rnn(x, self.hidden)
                self.hidden = h_n

            if self.fc_num == 1:
                return output, self.fc(output)
            else:
                return output, self.fc[step](output)
        
        elif self.fc_rnn == 'lstm':
            if restart:
                output, h_n = self.rnn(x, (torch.zeros(1, x.size(1), self.hidden_state_dim).cuda(), torch.zeros(1, x.size(1), self.hidden_state_dim).cuda()))
                self.hidden = h_n
            else:
                output, h_n = self.rnn(x, self.hidden)
                self.hidden = h_n

            if self.fc_num == 1:
                return output, self.fc[0](output)
            else:
                return output, self.fc[step](output)
        
        elif self.fc_rnn == 'transformer':
            if restart:
                self.hidden = torch.zeros(1, x.size(1), x.size(2)).cuda()

            self.hidden = torch.cat([self.hidden, x], 0)
            output = self.rnn(self.hidden)

            if self.fc_num == 1:
                return output, self.fc[0](output[0]).unsqueeze(0)
            else:
                return output, self.fc[step](output[0]).unsqueeze(0)

        elif self.fc_rnn == 'nfc':
            if restart:
                self.hidden = x
            else:
                self.hidden = torch.cat([self.hidden, x], 1)

            for i in range(self.fc_num):
                if self.hidden.size(1) == self.feature_num * (i+1):
                    return self.hidden, self.fc[i](self.hidden).unsqueeze(0)
                else:
                    print(self.hidden.size())
                    exit()

class Model(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super(Model, self).__init__()
        self.features = create_model(model_name, pretrained=pretrained, num_classes=0)

    def forward(self, x):
        out = self.features(x)
        print(out.shape)
        return out, out.detach()

# class Model(nn.Module):
#     def __init__(self, model_name, pretrained=False):
#         super(Model, self).__init__()

#         self.model = create_model(model_name, pretrained=pretrained, num_classes=1000)
#         self.features = nn.Sequential(*list(self.model.children())[:-1])
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#     def forward(self, x):
#         x = self.features(x)
#         out = self.avgpool(x).view(x.size(0), -1)
        
#         return out, out.detach() 
