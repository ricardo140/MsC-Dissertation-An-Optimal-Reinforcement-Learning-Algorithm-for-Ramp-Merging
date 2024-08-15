import torch
import torch.nn as nn
import gymnasium as gym
import pygame
import sys
import moviepy.editor as mpy
import matplotlib.pyplot as plt
import numpy as np
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

def draw_environment(state, screen, env, width, height):
    env.s = state
    img = env.render()
    img = pygame.surfarray.make_surface(img)
    img = pygame.transform.scale(img, (width, height))
    screen.blit(img, (0, 0))

def record_episode(env, model, video_filename="taxi_agent.mp4"):
    pygame.init()
    width, height = 400, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Taxi-V3")
    
    state = env.reset()[0]
    frames = []
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.int).unsqueeze(0)
            action = torch.argmax(model(state_tensor)).item()
            next_state, reward, done, _, _ = env.step(action)

            draw_environment(next_state, screen, env, width, height)

            frame = pygame.surfarray.array3d(screen)
            frames.append(frame)

            pygame.display.flip()

            if done:
                print("Succesful!")
                pygame.quit()
                break

            state = next_state

    clip = mpy.ImageSequenceClip(frames, fps=15)
    clip.write_videofile(video_filename)

def test_agent(env, model, num_episodes=100):
    total_rewards = []
    running_average = []

    total_reward = 0.0
    
    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.int).unsqueeze(0)
                action = torch.argmax(model(state_tensor)).item()
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                state = next_state
                steps += 1
            if steps % 100 == 0:
                done = True # entering in a loop

        total_reward += episode_reward
        total_rewards.append(episode_reward)
        running_average.append(np.mean(total_rewards[-100:]))
        print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward}")

    plt.plot(running_average)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Time')
    plt.savefig("dqn_taxi_test.png")

    return running_average


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    state = env.reset()[0]
    
    model = DQN(state_size=env.observation_space.n, action_size=env.action_space.n)
    model.load_state_dict(torch.load('best_taxi_model.pth'))
    
    print("We are going to test the loaded model")
    # record_episode(env, model)

    test_agent(env, model)