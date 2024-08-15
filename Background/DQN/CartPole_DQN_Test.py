import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import pygame 
import moviepy.editor as mpy
import sys
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)  
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def draw_environment(state, screen, env, width, height):
    env.s = state
    img = env.render()
    img = pygame.surfarray.make_surface(img)
    img = pygame.transform.scale(img, (width, height))
    screen.blit(img, (0, 0))

def record_episode(env, model, video_filename="cartpole-agent.mp4"):
    pygame.init()
    width, height = 400, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CartPole-v1")
    
    state = env.reset()[0]
    frames = []
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
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

    clip = mpy.ImageSequenceClip(frames, fps=120)
    clip.write_videofile(video_filename)

def test_agent(env, model, num_episodes=100):
    total_rewards = []
    running_average = []

    total_reward = 0.0
    
    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0.0
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                action = torch.argmax(model(state_tensor)).item()
                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if truncated:
                    done = True

        total_reward += episode_reward
        total_rewards.append(episode_reward)
        running_average.append(np.mean(total_rewards[-100:]))
        print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward}")

    plt.plot(running_average)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Time')
    plt.savefig("dqn_ct_test.png")

    return running_average

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    state = env.reset()[0]
    
    model = DQN(state_size=env.observation_space.shape[0], action_size=env.action_space.n)
    model.load_state_dict(torch.load('best_cartpole_model.pth'))
    
    print("We are going to test the loaded model")
    # record_episode(env, model)

    test_agent(env, model)