import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import namedtuple

gamma = 0.99
log_interval = 10
# class CustomTaxiEnv(gym.Env):
#     def __init__(self):
#         super(CustomTaxiEnv, self).__init__()
#         self.env = gym.make("Taxi-v3")

#     def step(self, action):
#         next_state, reward, done, info1, info2 = self.env.step(action)

#         # Modify rewards based on your custom values
#         if reward == 20:
#             reward = 1
#         elif reward == -10:
#             reward = -1
#         else:
#             reward = 0

#         return next_state, reward, done, info1, info2

#     def reset(self):
#         return self.env.reset()

#     def render(self, mode='human'):
#         return self.env.render(mode)

# env = CustomTaxiEnv()
env = gym.make("Taxi-v3")
state_size = env.observation_space.n
num_actions = env.action_space.n

SavedAction = namedtuple("SavedAction", ["log_prob", "value"])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        # self.emb = nn.Embedding(state_size, 4)
        self.fc1 = nn.Linear(state_size,256)
        self.fc2 = nn.Linear(256,256)
        self.actor = nn.Linear(256,num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.actor(x)
        return action_logits

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # self.emb = nn.Embedding(state_size, 4)
        self.fc1 = nn.Linear(state_size,256)
        self.fc2 = nn.Linear(256,256)
        self.critic = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        state_values = self.critic(x)
        return state_values

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.actor = Actor()
        # critic
        self.critic = Critic()
        # action and reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        action_prob = self.actor(x)
        state_values = self.critic(x)
        
        return action_prob, state_values

model = Policy()
eps = np.finfo(np.float32).eps.item()

epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995

def select_action(state, episode):
    epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
    
    state = torch.tensor(state, dtype = torch.long)
    state = F.one_hot(state, num_classes = state_size).float()
    logits, state_value = model(state)
    probs = F.softmax(logits, dim=-1)
    m = Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    model.saved_actions.append(SavedAction(log_prob, state_value))
    return action.item()

mean_value_loss = []
def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = []

    for r in model.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        entropy = -torch.sum(torch.exp(log_prob) * log_prob)
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    model.actor.optimizer.zero_grad()
    model.critic.optimizer.zero_grad()
    loss = (torch.stack(policy_losses).sum() + 1 * torch.stack(value_losses).sum())
    mean_value_loss.append(torch.stack(value_losses).mean().item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.actor.parameters(), 1)
    nn.utils.clip_grad_norm_(model.critic.parameters(), 1)
    model.actor.optimizer.step()
    model.critic.optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]

path = "C:/Users/ricar/OneDrive/Documentos/TFM/A2C/best_taxiA2C_model.pth"
def main():
    episodes = 1000
    r = []
    average = 0
    average_reward = []
    reward_episode = []
    total_steps = 0

    for episode in range(episodes):
        done = False
        total_reward = 0
        state = env.reset()[0]
        step = 0

        while not done:
            action = select_action(state, episode)
            state, reward, done, truncated, _ = env.step(action)
            step += 1
            total_steps += 1

            model.rewards.append(reward)
            total_reward += reward
            if total_steps % 200 == 0:
                finish_episode()
            if step % 200 == 0:
                done = True

        r.append(total_reward)
        prev_episodes = r[-100:]
        average = np.mean(prev_episodes)
        if average > 0:
            print(f"Solved Taxi with average reward: {average}")
            torch.save(model.state_dict(), path)
            break
        
        print(f"Episode: {episode}, total reward: {total_reward}, done: {done}, steps: {step}, average reward: {average}")
        average_reward.append(total_reward)
        if episode ==0:
            reward_episode.append(total_reward)
        else:
            smoothed_reward = reward_episode[-1]*0.9 + total_reward*0.1
            reward_episode.append(smoothed_reward)
    
    plt.figure()
    plt.plot(reward_episode, label = "score")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Smooth reward")
    plt.legend()
    plt.savefig("a2c_taxi.png")

if __name__ == "__main__":
    main()