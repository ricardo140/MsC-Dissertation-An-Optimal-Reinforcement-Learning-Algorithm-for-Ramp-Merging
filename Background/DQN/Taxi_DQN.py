import numpy as np 
import random
import torch 
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple
import matplotlib.pyplot as plt
import gymnasium as gym

Transition = namedtuple("Transition",('state', 'action', 'next_state', 'reward', 'done'))
# first we create the replay buffer that will contain the samples
class replayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, *args):
        self.memory.append(Transition(*args))
        if len(self.memory) > self.capacity:
            del self.memory[0]


    def sample(self, batch_size): # we want to give a random sample of the samples in the buffer
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)
    

# now we want to define the network architecture that we are going to use 
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.emb = nn.Embedding(state_size, 4)
        self.fc1 = nn.Linear(4, 128)  
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(self.emb(x)))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# we define the parameters
batch_size = 64
gamma = 0.95
initial_epsilon = 0.9
final_epsilon = 0.1
epsilon_decay = 0.001
learning_rate = 1e-3
tau = 0.2
# we will save the model with the best total reward so we can test it
best_total_reward = 1

env = gym.make("Taxi-v3").env
state_size = env.observation_space.n
action_size = env.action_space.n 

# we create the 2 netwroks we are going to use in this environment 
q_network = DQN(state_size, action_size)
target_network = DQN(state_size, action_size)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr= learning_rate, amsgrad= True)
buffer = replayBuffer(capacity=120000)

steps_done = 0
# creating a function to update the epsilon and select the next action by a given state
def select_action(state, epsilon):
    global steps_done
    sample = random.random()
    if sample < epsilon:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long)
        
    else:
        with torch.no_grad():
            q_values = q_network(state)
            return torch.argmax(q_values).view(1,1)
    
episode_durations = []

num_episodes = 1000
path = "C:/Users/ricar/OneDrive/Documentos/TFM/best_taxi_model.pth"
reward_episode = []
print(q_network)
for episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.int).unsqueeze(0)
    total_reward = 0
    n_ter = 0
    done = False

    while done == False:
        epsilon = max(final_epsilon, initial_epsilon - episode * epsilon_decay)
        action = select_action(state, epsilon)
        next_state, reward, done, info1, info2 = env.step(action.item())
        reward = torch.tensor([reward])
        next_state = torch.tensor(next_state, dtype=torch.int).unsqueeze(0)
        done = torch.tensor(done, dtype=torch.bool)

        buffer.push(state, action, next_state, reward, done)

        if len(buffer) >= batch_size:

            transitions = buffer.sample(batch_size)
            batch = Transition(*zip(*transitions))
            
            state_batch = torch.cat(batch.state)
            
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            done_batch = torch.stack(batch.done)
            next_state_batch = torch.cat(batch.next_state)
            predict_q_value = q_network(state_batch).gather(1, action_batch)
            
            next_state_value = target_network(next_state_batch).max(1)[0]
            expected_q_values = (~done_batch * next_state_value*gamma) + reward_batch

            loss = F.mse_loss(predict_q_value, expected_q_values.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            for param in q_network.parameters():
                param.grad.data.clamp_(-1,1)
            optimizer.step()

            target_update_freq = 1024
            if len(buffer)>=batch_size and steps_done%target_update_freq==0:
                target_state_dict = target_network.state_dict()
                q_state_dict = q_network.state_dict()
                for key in q_state_dict:
                    target_state_dict[key] = q_state_dict[key]*tau + target_state_dict[key]*(1-tau)

                target_network.load_state_dict(target_state_dict)

            if len(buffer) >= batch_size and steps_done % 7500 == 0:
                print(steps_done, loss.item())

        n_ter += 1
        if n_ter==7500:
            break    
        steps_done += 1
        state = next_state
        total_reward = total_reward + reward.item()
    
    if episode ==0:
        reward_episode.append(total_reward)
    else:
        smoothed_reward = reward_episode[-1]*0.9 + total_reward*0.1
        reward_episode.append(smoothed_reward)

    if total_reward > best_total_reward:
        print("Saving the model")
        best_total_reward = total_reward
        torch.save(q_network.state_dict(), path)
    if np.mean(reward_episode[-100:]) >= 0:
        print("Success")
        break 
    print(f"Episode: {episode}, total_reward: {total_reward}, done: {done}, epsilon: {epsilon}, steps: {n_ter}")

plt.figure(figsize=(10, 6))
plt.plot(reward_episode, label = "score")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.legend()
plt.savefig("dqn_taxi.png")