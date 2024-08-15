import gymnasium as gym 
import numpy as np 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import namedtuple

gamma = 0.99
log_interval = 10
env = gym.make("CartPole-v1")
state_size = 4
num_actions = env.action_space.n

SavedAction = namedtuple("SavedAction",["log_prob", "value"])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size,64)
        self.fc2 = nn.Linear(64,64)
        self.actor = nn.Linear(64,num_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_prob = F.softmax(self.actor(x), dim=1)
        return action_prob

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size,64)
        self.fc2 = nn.Linear(64,64)
        self.critic = nn.Linear(64,1)
    
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
optimizer = optim.Adam(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()

initial_temp = 1.0
final_temp = 0.1
temp_decay = 0.001
def select_action(state, temperature):
    state = torch.from_numpy(np.array([state])).float()
    probs, state_value = model(state)

    action_probs = F.softmax(probs/temperature, dim=1)
    m = Categorical(action_probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    action = action.item()

    return action

mean_value_loss = []
def finish_episode():
    rewards_tensor = torch.tensor(model.rewards[::-1], dtype=torch.float32)
    discounts = torch.tensor([gamma**i for i in range(len(rewards_tensor))], dtype=torch.float32)


    returns = torch.cumsum(rewards_tensor*discounts, dim=0)
    returns = torch.flip(returns, dims=[0])
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # calculate the losses and the advantage
    log_probs, values = zip(*model.saved_actions)
    log_probs = torch.stack(log_probs)
    values = torch.stack(values)
    advantages = returns - values.squeeze().detach()

    policy_losses = -log_probs * advantages.unsqueeze(1)
    value_losses = F.smooth_l1_loss(values.squeeze(), returns)
    mean_value_loss.append(value_losses.item())
    loss = policy_losses.mean() + value_losses 
    loss = loss.sum()

    optimizer.zero_grad()

    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1,1)
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]

path = "C:/Users/ricar/OneDrive/Documentos/TFM/A2C/best_cartpoleA2C_model.pth"
def main():
    episodes = 1000
    r = []
    average = 0
    average_reward = []
    reward_episode = []
    for episode in range(episodes):
        done = False
        total_reward = 0
        state, _ = env.reset()

        while not done:
            temp = max(final_temp, initial_temp - episode*temp_decay)
            action = select_action(state, temp)
            state, reward, done, truncated, _ = env.step(action)

            model.rewards.append(reward)
            total_reward += reward
            if truncated:
                done = True
        
        r.append(total_reward)


        prev_episodes = r[-100:]
        average = np.mean(prev_episodes)
        if average > 195:
            print(f"Solved CartPole with average reward: {average}")
            torch.save(model.state_dict(), path)
            break
        finish_episode()

        print(f"Episode: {episode}, total reward: {total_reward}, done: {done}, mean: {average}")

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
    plt.savefig("a2c_cartpole.png")

if __name__ == "__main__":
    main()