import random

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from itertools import product
# networks architectures: Actor-Critic

class ActorNN(nn.Module):
    """
    Actor Neural Network architecture for PPO
    """
    def __init__(self, input_dim, output_dim, alpha,num_cells, dropout_prob=0.5):
        super(ActorNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_cells)
        self.fc2 = nn.Linear(num_cells, num_cells)
        self.fc3 = nn.Linear(num_cells, output_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
    
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        action_prob = F.softmax(self.fc3(x), dim=-1)
        return action_prob

class CriticNN(nn.Module):
    """
    Critic Neural Network architecture for PPO
    """
    def __init__(self, input_dim, alpha, num_cells, dropout_prob=0.5):
        super(CriticNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, num_cells)
        self.fc2 = nn.Linear(num_cells, num_cells)
        self.fc3 = nn.Linear(num_cells, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        val = self.fc3(x)
        return val

# we create a buffer to store the information from the environment

class PPOBuffer():
    """
    Replay buffer for PPO
    """
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.vals = []
        self.is_terminals = []

        self.batch_size = batch_size

    def generate_batches(self):
        # generates batches of states, actions, logprobs, rewards, vals, is_terminals
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int32)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.logprobs),\
                np.array(self.rewards),\
                np.array(self.vals),\
                np.array(self.is_terminals),\
                batches
    
    def store(self, state, action, logprob, reward, val, done):
        # stores the states, actions, logprobs, rewards, vals, is_terminals
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.vals.append(val)
        self.is_terminals.append(done)

    def restore(self):
        # restores the states, actions, logprobs, rewards, vals, is_terminals
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.vals = []
        self.is_terminals = []

# now lets create the agent that is going to learn from the environment

class Agent():
    """"
    Agent class to learn from the environment
    """
    def __init__(self, discount_factor, clip, lamda, alpha,
                epochs, batch_size, state_n, action_n, num_cells):

        self.discount_factor = discount_factor
        self.clip = clip
        self.lamda = lamda
        self.epochs = epochs
        self.state_n = state_n

        self.actor = ActorNN(state_n, action_n, alpha[0], num_cells)
        self.critic = CriticNN(state_n, alpha[1], num_cells)
        self.memory = PPOBuffer(batch_size)

    def store(self, state, action, logprob, reward, val, done):
        # stores the states, actions, logprobs, rewards, vals, is_terminals
        self.memory.store(state, action, logprob, reward, val, done)

    def take_action(self, state):
    # takes an action based on the state
        state = torch.tensor(state, dtype = torch.long)
        state = F.one_hot(state, num_classes = self.state_n).float()
        
        # we calculate the action probs using the Actor NN
        action_prob = self.actor(state)
        dist = Categorical(action_prob)
        action = dist.sample()
        log_probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()

        # we calculate the value using the Critic NN
        value = self.critic(state)
        value = torch.squeeze(value).item()

        return action, log_probs, value
    
    def advantage(self, rewards, values, is_terminals):
        # we calculate the advantage
        steps = len(rewards)
        advantage = np.zeros(steps)
        gae = 0
        
        for step in reversed(range(steps - 1)):
            delta = rewards[step] + self.discount_factor * values[step + 1] * (1 - is_terminals[step]) - values[step]
            gae = delta + self.discount_factor * self.lamda * (1 - is_terminals[step]) * gae
            advantage[step] = gae
        
        advantage = torch.tensor(advantage, dtype=torch.float32)
        return advantage
    
    def learn(self, max_grad_norm):
        # Let the agent learn from the buffer
        for _ in range(self.epochs):
            states_full, actions_full, logprobs_full, rewards, vals, is_terminals, batches = self.memory.generate_batches()

            advantage_array = self.advantage(rewards, vals, is_terminals)
            values = torch.tensor(vals, dtype=torch.float32)
            advantage_array = (advantage_array - advantage_array.mean()) / (advantage_array.std() + 1e-10)  # Normalization

            for batch in batches:
                states = torch.tensor(states_full[batch], dtype=torch.long)
                states = F.one_hot(states, num_classes=self.state_n).float()
                old_probs = torch.tensor(logprobs_full[batch], dtype=torch.float32)
                actions = torch.tensor(actions_full[batch], dtype=torch.long)

                # Calculate action probabilities and critic value
                dist = Categorical(self.actor(states))
                value = self.critic(states).squeeze()

                # Calculate the ratio
                new_probs = dist.log_prob(actions)
                ratio = new_probs.exp() / old_probs.exp()  # Convert to exponentials
                # print(advantage_array[batch], ratio.shape)
                # Clip the ratio
                weight = advantage_array[batch] * ratio
                clipped_weight = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantage_array[batch]

                # Choose the minimum between the clipped weight and the weight and calculate losses
                loss = -torch.min(weight, clipped_weight).mean()

                critic_loss = F.mse_loss(value, values[batch])

                total_loss = loss + critic_loss # - 0.01 * dist.entropy().mean() # Include coefficient for critic loss

                # Optimization
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        
        # Clear the buffer after each epoch
        self.memory.restore()

        return loss, critic_loss

class testAgent():
    """
    Tests the saved model
    """
    def __init__(self, state_n, action_n, actor_path, critic_path, num_cells):
        self.state_n = state_n
        self.actor = ActorNN(state_n, action_n, num_cells = num_cells, alpha=0) 
        self.critic = CriticNN(state_n, num_cells=num_cells, alpha=0) 
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.actor.eval()
        self.critic.eval()

    def take_action(self, state):
        # takes an action based on the state
        state = torch.tensor(state, dtype = torch.long)
        state = F.one_hot(state, num_classes = self.state_n).float()
        action_prob = self.actor(state)
        # action = torch.argmax(action_prob)
        dist = Categorical(action_prob)
        action = dist.sample()
        action = torch.squeeze(action).item()
        return action

def test_single_episode(agent, env, video_dir):
    env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda x: True)
    
    state = env.reset()[0]
    done = False
    score = 0
    while not done:
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = 1 if (terminated or truncated) else 0
        score += reward
        state = next_state

    env.close()
    return score

def test(num_episodes, model):
    score_hist = []
    average_reward = []
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        score = 0
        steps = 0
        while not done:
            steps += 1
            action = model.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = 1 if (terminated or truncated) else 0
            score += reward
            state = next_state
        score_hist.append(score)
        average_reward.append(np.mean(score_hist[-100:]))
    return average_reward

# now we train the model and define parameters

seed = 43201
env = gym.make("Taxi-v3", render_mode="rgb_array")
env.np_random = np.random.default_rng(seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

gamma = 0.99
clip = 0.3
lamda = 0.95
state_n = env.observation_space.n
action_n = env.action_space.n

batch_size_range = [64]
alpha_range = [[1e-3, 1e-3]]
epochs_range = [5]
num_cells_range = [256]
memory_size_range = [8192]
max_grad_norm = 1

param_combinations = list(product(batch_size_range, alpha_range, epochs_range, num_cells_range, memory_size_range))
best_average_reward = -float('inf')
best_params = None

# for (batch_size, alpha, epochs, num_cells, memory_size) in param_combinations:
#     print(f"Training with batch_size={batch_size}, alpha={alpha}, epochs={epochs}, num_cells={num_cells}, memory_size={memory_size}")

#     agent = Agent(gamma, clip, lamda, alpha, epochs, batch_size, state_n, action_n, num_cells)
#     num_episodes = 2500
#     actor_loss = []
#     critic_loss = []
#     score_hist = []
#     learn_steps = 0
#     average_reward = 0
#     total_steps = 0
#     best_score = 0
#     reward_episode = []

#     for episode in range(num_episodes):
#         state = env.reset()[0]
#         terminated, truncated = False, False
#         done = False
#         score = 0
#         episode_steps = 0
#         while not done:
#             episode_steps += 1
#             action, probs, val = agent.take_action(state)
#             next_state, reward, terminated, _, _ = env.step(action)
#             done = 1 if (terminated) else 0
#             total_steps += 1
#             score += reward 
#             agent.store(state, action, probs, reward, val, done)

#             if total_steps % memory_size == 0:
#                 actorloss, criticloss = agent.learn(max_grad_norm)
#                 actor_loss.append(actorloss.item())
#                 critic_loss.append(criticloss.item())
#             state = next_state

#             if episode_steps % 500 == 0:
#                 break
#         score_hist.append(score)
#         score_hist.append(score)
#         if episode == 0:
#             reward_episode.append(score)
#         else:
#             reward_episode.append(reward_episode[-1]*0.9 + score*0.1)
#         average_reward = np.mean(score_hist[-100:])
#         print(f"Episode: {episode}, score: {score}, average reward: {average_reward}, total_steps: {episode_steps}")

#         if score >= 12:
#             print(f"Saving model at episode {episode} with average reward {average_reward}")
#             torch.save(agent.actor.state_dict(), 'ppo_actor.pth')
#             torch.save(agent.critic.state_dict(), 'ppo_critic.pth')

#         if average_reward >= 0:
#             break

#     if average_reward > best_average_reward:
#         best_average_reward = average_reward
#         best_params = (batch_size, alpha, epochs, num_cells, memory_size)
#         print(f"New best average reward: {best_average_reward}, with params: {best_params}")

# print(f"Best average reward: {best_average_reward} with params: {best_params}")
# fig = plt.figure()
# plt.plot(reward_episode, label = "score")
# plt.xlabel("episode")
# plt.ylabel("score")
# plt.title("Evolution of score during training")
# plt.legend()
# fig.savefig("ppo_score.png")

# fig = plt.figure(figsize=(10, 6))
# plt.plot(actor_loss, color = "red", label = "score")
# plt.xlabel("episode")
# plt.ylabel("loss")
# plt.title("Evolution of actor_loss during training")
# plt.legend()
# fig.savefig("taxi_model/ppo_actor_loss.png")

# fig = plt.figure(figsize=(10, 6))
# plt.plot(critic_loss, color = "red", label = "score")
# plt.xlabel("episode")
# plt.ylabel("loss")
# plt.title("Evolution of critic_loss during training")
# plt.legend()
# fig.savefig("taxi_model/ppo_critic_loss.png")

# we test the model 
actor_path = "ppo_actor.pth"
critic_path = "ppo_critic.pth"
agent = testAgent(state_n, action_n, actor_path, critic_path, 256)
env.reset()
reward = test(100, agent)
fig = plt.figure()
plt.plot(reward)
plt.xlabel("Episode")
plt.ylabel("Average reward")
plt.title("Average reward over time")
fig.savefig("ppo_test_taxi.png")

# test_single_episode(agent, env, "taxi_model/video")
