import gymnasium as gym
import numpy as np
import torch
from AC_cartpole import Policy 
import pygame
import sys
import moviepy.editor as mpy
import matplotlib.pyplot as plt

path = "C:/Users/ricar/OneDrive/Documentos/TFM/A2C/best_cartpoleA2C_model.pth"

def select_action_test(state):
    state = torch.from_numpy(np.array([state])).float()
    probs, _ = model(state)
    action = torch.argmax(probs).item()
    return action
def draw_environment(state, screen, env, width, height):
    env.s = state
    img = env.render()
    img = pygame.surfarray.make_surface(img)
    img = pygame.transform.scale(img, (width, height))
    screen.blit(img, (0, 0))

def test_record(model, env, video_filename="cartpole_test.mp4"):
    state, _ = env.reset()
    done = False
    frames = []

    while not done:
        with torch.no_grad():
            action = select_action_test(state)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state

            # Render the environment and store the frame
            frame = env.render()
            frames.append(frame)

    # Create a video from the frames
    clip = mpy.ImageSequenceClip(frames, fps=120)
    clip.write_videofile(video_filename)

    print(f"Video saved as {video_filename}")
def test_model(model, num_episodes=100):
    total_rewards = []
    running_average = []

    total_reward = 0.0
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        while not done:
            with torch.no_grad():
                action = select_action_test(state)
                next_state, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                state = next_state
                steps += 1
                if steps % 500 == 0: # do not know why truncated here does not work, we do it by hand
                    done = True

        total_reward += episode_reward
        total_rewards.append(episode_reward)
        running_average.append(np.mean(total_rewards[-100:]))
        print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward}")

    plt.plot(running_average)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward Over Time')
    plt.savefig("a2c_ct_test.png")

    return running_average

if __name__ == "__main__":
    # Load the trained model
    model = Policy()
    model.load_state_dict(torch.load(path))

    # Set the model in evaluation mode
    model.eval()

    # Create CartPole environment
    env = gym.make("CartPole-v1", render_mode="rgb_array").env

    # Test the model
    num_test_episodes = 100
    average_reward = test_model(model, num_test_episodes)

    print("We are going to test the loaded model")
    # test_record(model, env)
