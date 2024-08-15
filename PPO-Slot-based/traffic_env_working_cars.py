import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
# this environment is rendered as an array

class HighwayEnv(gym.Env):
    # for now safety distance is considered just 1 as representing 1 length slot
    def __init__(self, num_lanes=3, lane_length=5, on_ramp_length = 100, slot_length = 20):
        super(HighwayEnv, self).__init__()

        self.num_lanes = num_lanes
        self.lane_length = lane_length
        self.observation_space = spaces.Box(low = 0, high = 1, shape = (self.num_lanes, self.lane_length-1), dtype=np.int32)
        # the action space is a tuple (position as 0-end, action 0-4)
        self.action_space = spaces.Discrete(self.num_lanes * (self.lane_length-1) * 5)
        self.slot_length = slot_length
        self.on_ramp_length = on_ramp_length
        self.positions = np.arange(0, self.num_lanes * (self.lane_length-1)).reshape((self.num_lanes, self.lane_length-1))

    def get_reward(self, collision, done, position, act, current_state):
        """
        This function computes the total reward given by taking the action, penalises collisions
        and tyring to move a free slot. Rewards are positive if we move a non-free slot to a free slot

        args:
        action: (position, action)
        current_state: (numpy array) the current postions on the road
        done: bool indicating if the episode is done

        returns:
        int: the total reward
        """
        reward = 0

        if (self.target_slot <= 2) and current_state[2][self.target_slot+2] == 1 \
            and self.road[2][self.target_slot+1] == 0:
                reward += 50
        elif (self.target_slot <= 2) and current_state[2][self.target_slot+2] == 0 \
            and self.road[2][self.target_slot+1] == 1:
                reward += -50
        if done:
            if collision:
                reward += -10

            else:
                if self.road[self.num_lanes-1][self.target_slot+1] == 0:
                    reward += 50
                else:
                    reward += -0.1
        else:
            # if (self.target_slot <= 2) and (lane, pos) == (2, self.target_slot + 2):
            #     if (current_state[lane][pos] == 1) and (act in target_actions):
            #         return 50
            #     else:
            #         return 0
                reward += 0.1

        return reward

    def encoding(self, action):
        """
        This function decodes the action into (pos,action)
        """
        pos = int(np.ceil((action - 4) / 5))
        act = int(action - (5 * pos))

        return (pos, act)

    def calculate_target_slot(self):
        """
        This function calculates the target slot that we want to check (free or not)
        """
        # we define the target slot as the one that will arrive on time to merge the
        # vehicle on the merging lane.
        self.target_slot = 7

    def step(self, action):
        """
        This function performs the step of the environment and takes the actions

        args:
        action: (position, action)
        """

        pos, act = self.encoding(action)
        # print(pos, act)
        lane = np.where(self.positions == pos)[0]
        pos = np.where(self.positions == pos)[1] # to include the front column
        current_state = self.road.copy()

        if current_state[lane, pos] == 1: # vehicle at slot (lane, pos)
            if act == 1: # forward
                if pos - 1 >= 0:
                    self.road[lane, pos - 1] = current_state[lane, pos]
                self.road[lane, pos] = 0 # we consider taking the action

            elif act == 2: # backward
                if pos + 1 < self.lane_length:
                    self.road[lane, pos + 1] = current_state[lane, pos]
                self.road[lane, pos] = 0 # we consider taking the action

            elif act == 3 and lane - 1 >= 0: # forward right
                if pos - 1 >= 0:
                    self.road[lane - 1, pos - 1] = current_state[lane, pos]
                self.road[lane, pos] = 0

            elif act == 4 and lane + 1 < self.num_lanes: # forward left
                if pos - 1 >= 0:
                    self.road[lane + 1, pos - 1] = current_state[lane, pos]
                self.road[lane, pos] = 0

        collision = self.check_collision(action, current_state)
        self.movement()
        self.target_slot -= 1

        done = self.check_done(collision)
        reward = self.get_reward(collision, done, (lane, pos), act, current_state)

        return self.window, reward, done, {"positions": self.positions}

    def check_collision(self, action, current_state):
        """
        This function checks if there is a collision
        """
        pos, act = self.encoding(action)
        lane = np.where(self.positions == pos)[0]
        pos = np.where(self.positions == pos)[1]

        if current_state[lane, pos] == 1:
            if act == 1 and pos - 1 >= 0 and\
                current_state[lane, pos-1] == current_state[lane, pos]:
                # collision
                return True

            elif act == 2 and pos + 1 < self.lane_length and\
                current_state[lane, pos+1] == current_state[lane, pos]:
                # collision
                return True

            elif act == 3 and lane - 1 >= 0 and pos-1>=0:
                if current_state[lane - 1, pos - 1] == current_state[lane, pos]:
                # collision
                    return True

            elif act == 4 and lane + 1 < self.num_lanes and pos-1 >=0:
                if current_state[lane + 1, pos - 1] == current_state[lane, pos]:
                # collision
                    return True

        return False

    def check_done(self, collision):
        """ checks if the episode has finished, the environment
        finishes when a free slot reaches the target lane at the
        merging point. """
        # we will assume that the merging point is position (2,1)
        if collision:
            return True
        else:
            if self.target_slot == 0:
                return True
            return False

    def reset(self):
        """ resets the environment """
        self.create_road()

        self.calculate_target_slot()
        return self.window

    def movement(self):
        """This function simulates the movement of the road at a certain speed"""
        # we move it to the left so it moves forward the lane as simulating the vehicules moving
        # forward all at the same speed
        self.road[:,:-1] = self.road[:,1:]
        new_column = np.random.binomial(n=1, size=(self.num_lanes, 1), p = np.random.random())
        self.road[:,-1:] = new_column
        # if self.target_slot - 1 == self.lane_length-1:
        #     self.road[self.num_lanes-1, self.target_slot-1] = 1
        #     self.road[self.num_lanes-2, self.target_slot-1] = 1
        # if self.target_slot == self.lane_length-1:
        #         self.road[self.num_lanes-1, self.target_slot] = 1
        # if self.target_slot+1 == self.lane_length-1:
        #     self.road[self.num_lanes-1, self.target_slot+1] = 1

    def create_road(self):
        """
        This function creates the road with 3 lanes all with 0s representing the slots
        """
        prob = np.random.random()
        self.road = np.random.binomial(n=1, size=(self.num_lanes, self.lane_length), p = prob)
        self.window = self.road[:,:4]

    def render(self):
            """Render the road"""
            print("------------------------")
            for i in range(self.num_lanes):
                print(self.road[i])
                print("------------------------")

def test_environment(env, num_episodes=1):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()  # Take a random action
            # if env.target_slot == 2:
            #     action = 56
            # if env.target_slot == 1:
            #     action = 47
            # action = 0
            env.render()
            # print(env.target_slot)
            next_state, reward, done, info = env.step(action)
            print(action)
            print(reward)
            print(next_state)
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

if __name__ == "__main__":
    env = HighwayEnv()
    # seed = 123
    # np.random.seed(seed)
    # random.seed(seed)
    # env.observation_space.seed(seed)
    test_environment(env)

class LightHighwayEnv(gym.Env):
    # for now safety distance is considered just 1 as representing 1 length slot
    def __init__(self, num_lanes=3, lane_length=5, on_ramp_length = 100, slot_length = 20):
        super(LightHighwayEnv, self).__init__()

        self.num_lanes = num_lanes
        self.lane_length = lane_length
        self.observation_space = spaces.Box(low = 0, high = 1, shape = (self.num_lanes, self.lane_length-1), dtype=np.int32)
        # the action space is a tuple (position as 0-end, action 0-4)
        self.action_space = spaces.Discrete(self.num_lanes * (self.lane_length-1) * 5)
        self.slot_length = slot_length
        self.on_ramp_length = on_ramp_length
        self.positions = np.arange(0, self.num_lanes * (self.lane_length-1)).reshape((self.num_lanes, self.lane_length-1))

    def get_reward(self, collision, done, position, act, current_state):
        """
        This function computes the total reward given by taking the action, penalises collisions
        and tyring to move a free slot. Rewards are positive if we move a non-free slot to a free slot

        args:
        action: (position, action)
        current_state: (numpy array) the current postions on the road
        done: bool indicating if the episode is done

        returns:
        int: the total reward
        """
        reward = 0

        if (self.target_slot <= 2) and current_state[2][self.target_slot+2] == 1 \
            and self.road[2][self.target_slot+1] == 0:
                reward += 50
        elif (self.target_slot <= 2) and current_state[2][self.target_slot+2] == 0 \
            and self.road[2][self.target_slot+1] == 1:
                reward += -50
        if done:
            if collision:
                reward += -10

            else:
                if self.road[self.num_lanes-1][self.target_slot+1] == 0:
                    reward += 50
                else:
                    reward += -0.1
        else:
            # if (self.target_slot <= 2) and (lane, pos) == (2, self.target_slot + 2):
            #     if (current_state[lane][pos] == 1) and (act in target_actions):
            #         return 50
            #     else:
            #         return 0
                reward += 0.1

        return reward

    def encoding(self, action):
        """
        This function decodes the action into (pos,action)
        """
        pos = int(np.ceil((action - 4) / 5))
        act = int(action - (5 * pos))

        return (pos, act)

    def calculate_target_slot(self):
        """
        This function calculates the target slot that we want to check (free or not)
        """
        # we define the target slot as the one that will arrive on time to merge the
        # vehicle on the merging lane.
        self.target_slot = 7

    def step(self, action):
        """
        This function performs the step of the environment and takes the actions

        args:
        action: (position, action)
        """

        pos, act = self.encoding(action)
        # print(pos, act)
        lane = np.where(self.positions == pos)[0]
        pos = np.where(self.positions == pos)[1] # to include the front column
        current_state = self.road.copy()

        if current_state[lane, pos] == 1: # vehicle at slot (lane, pos)
            if act == 1: # forward
                if pos - 1 >= 0:
                    self.road[lane, pos - 1] = current_state[lane, pos]
                self.road[lane, pos] = 0 # we consider taking the action

            elif act == 2: # backward
                if pos + 1 < self.lane_length:
                    self.road[lane, pos + 1] = current_state[lane, pos]
                self.road[lane, pos] = 0 # we consider taking the action

            elif act == 3 and lane - 1 >= 0: # forward right
                if pos - 1 >= 0:
                    self.road[lane - 1, pos - 1] = current_state[lane, pos]
                self.road[lane, pos] = 0

            elif act == 4 and lane + 1 < self.num_lanes: # forward left
                if pos - 1 >= 0:
                    self.road[lane + 1, pos - 1] = current_state[lane, pos]
                self.road[lane, pos] = 0

        collision = self.check_collision(action, current_state)
        self.movement()
        self.target_slot -= 1

        done = self.check_done(collision)
        reward = self.get_reward(collision, done, (lane, pos), act, current_state)

        return self.window, reward, done, {"positions": self.positions}

    def check_collision(self, action, current_state):
        """
        This function checks if there is a collision
        """
        pos, act = self.encoding(action)
        lane = np.where(self.positions == pos)[0]
        pos = np.where(self.positions == pos)[1]

        if current_state[lane, pos] == 1:
            if act == 1 and pos - 1 >= 0 and\
                current_state[lane, pos-1] == current_state[lane, pos]:
                # collision
                return True

            elif act == 2 and pos + 1 < self.lane_length and\
                current_state[lane, pos+1] == current_state[lane, pos]:
                # collision
                return True

            elif act == 3 and lane - 1 >= 0 and pos-1>=0:
                if current_state[lane - 1, pos - 1] == current_state[lane, pos]:
                # collision
                    return True

            elif act == 4 and lane + 1 < self.num_lanes and pos-1 >=0:
                if current_state[lane + 1, pos - 1] == current_state[lane, pos]:
                # collision
                    return True

        return False

    def check_done(self, collision):
        """ checks if the episode has finished, the environment
        finishes when a free slot reaches the target lane at the
        merging point. """
        # we will assume that the merging point is position (2,1)
        if collision:
            return True
        else:
            if self.target_slot == 0:
                return True
            return False

    def reset(self):
        """ resets the environment """
        self.create_road()

        self.calculate_target_slot()
        return self.window

    def movement(self):
        """This function simulates the movement of the road at a certain speed"""
        # we move it to the left so it moves forward the lane as simulating the vehicules moving
        # forward all at the same speed
        self.road[:,:-1] = self.road[:,1:]
        new_column = np.random.binomial(n=1, size=(self.num_lanes, 1), p = 0.3)
        self.road[:,-1:] = new_column
        # if self.target_slot - 1 == self.lane_length-1:
        #     self.road[self.num_lanes-1, self.target_slot-1] = 1
        #     self.road[self.num_lanes-2, self.target_slot-1] = 1
        # if self.target_slot == self.lane_length-1:
        #         self.road[self.num_lanes-1, self.target_slot] = 1
        # if self.target_slot+1 == self.lane_length-1:
        #     self.road[self.num_lanes-1, self.target_slot+1] = 1

    def create_road(self):
        """
        This function creates the road with 3 lanes all with 0s representing the slots
        """
        prob = np.random.random()
        self.road = np.random.binomial(n=1, size=(self.num_lanes, self.lane_length), p = 0.3)
        self.window = self.road[:,:4]

    def render(self):
            """Render the road"""
            print("------------------------")
            for i in range(self.num_lanes):
                print(self.road[i])
                print("------------------------")

class MediumHighwayEnv(gym.Env):
    # for now safety distance is considered just 1 as representing 1 length slot
    def __init__(self, num_lanes=3, lane_length=5, on_ramp_length = 100, slot_length = 20):
        super(MediumHighwayEnv, self).__init__()

        self.num_lanes = num_lanes
        self.lane_length = lane_length
        self.observation_space = spaces.Box(low = 0, high = 1, shape = (self.num_lanes, self.lane_length-1), dtype=np.int32)
        # the action space is a tuple (position as 0-end, action 0-4)
        self.action_space = spaces.Discrete(self.num_lanes * (self.lane_length-1) * 5)
        self.slot_length = slot_length
        self.on_ramp_length = on_ramp_length
        self.positions = np.arange(0, self.num_lanes * (self.lane_length-1)).reshape((self.num_lanes, self.lane_length-1))

    def get_reward(self, collision, done, position, act, current_state):
        """
        This function computes the total reward given by taking the action, penalises collisions
        and tyring to move a free slot. Rewards are positive if we move a non-free slot to a free slot

        args:
        action: (position, action)
        current_state: (numpy array) the current postions on the road
        done: bool indicating if the episode is done

        returns:
        int: the total reward
        """
        reward = 0

        if (self.target_slot <= 2) and current_state[2][self.target_slot+2] == 1 \
            and self.road[2][self.target_slot+1] == 0:
                reward += 50
        elif (self.target_slot <= 2) and current_state[2][self.target_slot+2] == 0 \
            and self.road[2][self.target_slot+1] == 1:
                reward += -50
        if done:
            if collision:
                reward += -10

            else:
                if self.road[self.num_lanes-1][self.target_slot+1] == 0:
                    reward += 50
                else:
                    reward += -0.1
        else:
            # if (self.target_slot <= 2) and (lane, pos) == (2, self.target_slot + 2):
            #     if (current_state[lane][pos] == 1) and (act in target_actions):
            #         return 50
            #     else:
            #         return 0
                reward += 0.1

        return reward

    def encoding(self, action):
        """
        This function decodes the action into (pos,action)
        """
        pos = int(np.ceil((action - 4) / 5))
        act = int(action - (5 * pos))

        return (pos, act)

    def calculate_target_slot(self):
        """
        This function calculates the target slot that we want to check (free or not)
        """
        # we define the target slot as the one that will arrive on time to merge the
        # vehicle on the merging lane.
        self.target_slot = 7

    def step(self, action):
        """
        This function performs the step of the environment and takes the actions

        args:
        action: (position, action)
        """

        pos, act = self.encoding(action)
        # print(pos, act)
        lane = np.where(self.positions == pos)[0]
        pos = np.where(self.positions == pos)[1] # to include the front column
        current_state = self.road.copy()

        if current_state[lane, pos] == 1: # vehicle at slot (lane, pos)
            if act == 1: # forward
                if pos - 1 >= 0:
                    self.road[lane, pos - 1] = current_state[lane, pos]
                self.road[lane, pos] = 0 # we consider taking the action

            elif act == 2: # backward
                if pos + 1 < self.lane_length:
                    self.road[lane, pos + 1] = current_state[lane, pos]
                self.road[lane, pos] = 0 # we consider taking the action

            elif act == 3 and lane - 1 >= 0: # forward right
                if pos - 1 >= 0:
                    self.road[lane - 1, pos - 1] = current_state[lane, pos]
                self.road[lane, pos] = 0

            elif act == 4 and lane + 1 < self.num_lanes: # forward left
                if pos - 1 >= 0:
                    self.road[lane + 1, pos - 1] = current_state[lane, pos]
                self.road[lane, pos] = 0

        collision = self.check_collision(action, current_state)
        self.movement()
        self.target_slot -= 1

        done = self.check_done(collision)
        reward = self.get_reward(collision, done, (lane, pos), act, current_state)

        return self.window, reward, done, {"positions": self.positions}

    def check_collision(self, action, current_state):
        """
        This function checks if there is a collision
        """
        pos, act = self.encoding(action)
        lane = np.where(self.positions == pos)[0]
        pos = np.where(self.positions == pos)[1]

        if current_state[lane, pos] == 1:
            if act == 1 and pos - 1 >= 0 and\
                current_state[lane, pos-1] == current_state[lane, pos]:
                # collision
                return True

            elif act == 2 and pos + 1 < self.lane_length and\
                current_state[lane, pos+1] == current_state[lane, pos]:
                # collision
                return True

            elif act == 3 and lane - 1 >= 0 and pos-1>=0:
                if current_state[lane - 1, pos - 1] == current_state[lane, pos]:
                # collision
                    return True

            elif act == 4 and lane + 1 < self.num_lanes and pos-1 >=0:
                if current_state[lane + 1, pos - 1] == current_state[lane, pos]:
                # collision
                    return True

        return False

    def check_done(self, collision):
        """ checks if the episode has finished, the environment
        finishes when a free slot reaches the target lane at the
        merging point. """
        # we will assume that the merging point is position (2,1)
        if collision:
            return True
        else:
            if self.target_slot == 0:
                return True
            return False

    def reset(self):
        """ resets the environment """
        self.create_road()

        self.calculate_target_slot()
        return self.window

    def movement(self):
        """This function simulates the movement of the road at a certain speed"""
        # we move it to the left so it moves forward the lane as simulating the vehicules moving
        # forward all at the same speed
        self.road[:,:-1] = self.road[:,1:]
        new_column = np.random.binomial(n=1, size=(self.num_lanes, 1), p = 0.5)
        self.road[:,-1:] = new_column
        # if self.target_slot - 1 == self.lane_length-1:
        #     self.road[self.num_lanes-1, self.target_slot-1] = 1
        #     self.road[self.num_lanes-2, self.target_slot-1] = 1
        # if self.target_slot == self.lane_length-1:
        #         self.road[self.num_lanes-1, self.target_slot] = 1
        # if self.target_slot+1 == self.lane_length-1:
        #     self.road[self.num_lanes-1, self.target_slot+1] = 1

    def create_road(self):
        """
        This function creates the road with 3 lanes all with 0s representing the slots
        """
        prob = np.random.random()
        self.road = np.random.binomial(n=1, size=(self.num_lanes, self.lane_length), p = 0.5)
        self.window = self.road[:,:4]

    def render(self):
            """Render the road"""
            print("------------------------")
            for i in range(self.num_lanes):
                print(self.road[i])
                print("------------------------")

class HighHighwayEnv(gym.Env):
    # for now safety distance is considered just 1 as representing 1 length slot
    def __init__(self, num_lanes=3, lane_length=5, on_ramp_length = 100, slot_length = 20):
        super(HighHighwayEnv, self).__init__()

        self.num_lanes = num_lanes
        self.lane_length = lane_length
        self.observation_space = spaces.Box(low = 0, high = 1, shape = (self.num_lanes, self.lane_length-1), dtype=np.int32)
        # the action space is a tuple (position as 0-end, action 0-4)
        self.action_space = spaces.Discrete(self.num_lanes * (self.lane_length-1) * 5)
        self.slot_length = slot_length
        self.on_ramp_length = on_ramp_length
        self.positions = np.arange(0, self.num_lanes * (self.lane_length-1)).reshape((self.num_lanes, self.lane_length-1))

    def get_reward(self, collision, done, position, act, current_state):
        """
        This function computes the total reward given by taking the action, penalises collisions
        and tyring to move a free slot. Rewards are positive if we move a non-free slot to a free slot

        args:
        action: (position, action)
        current_state: (numpy array) the current postions on the road
        done: bool indicating if the episode is done

        returns:
        int: the total reward
        """
        reward = 0

        if (self.target_slot <= 2) and current_state[2][self.target_slot+2] == 1 \
            and self.road[2][self.target_slot+1] == 0:
                reward += 50
        elif (self.target_slot <= 2) and current_state[2][self.target_slot+2] == 0 \
            and self.road[2][self.target_slot+1] == 1:
                reward += -50
        if done:
            if collision:
                reward += -10

            else:
                if self.road[self.num_lanes-1][self.target_slot+1] == 0:
                    reward += 50
                else:
                    reward += -0.1
        else:
            # if (self.target_slot <= 2) and (lane, pos) == (2, self.target_slot + 2):
            #     if (current_state[lane][pos] == 1) and (act in target_actions):
            #         return 50
            #     else:
            #         return 0
                reward += 0.1

        return reward

    def encoding(self, action):
        """
        This function decodes the action into (pos,action)
        """
        pos = int(np.ceil((action - 4) / 5))
        act = int(action - (5 * pos))

        return (pos, act)

    def calculate_target_slot(self):
        """
        This function calculates the target slot that we want to check (free or not)
        """
        # we define the target slot as the one that will arrive on time to merge the
        # vehicle on the merging lane.
        self.target_slot = 7

    def step(self, action):
        """
        This function performs the step of the environment and takes the actions

        args:
        action: (position, action)
        """

        pos, act = self.encoding(action)
        # print(pos, act)
        lane = np.where(self.positions == pos)[0]
        pos = np.where(self.positions == pos)[1] # to include the front column
        current_state = self.road.copy()

        if current_state[lane, pos] == 1: # vehicle at slot (lane, pos)
            if act == 1: # forward
                if pos - 1 >= 0:
                    self.road[lane, pos - 1] = current_state[lane, pos]
                self.road[lane, pos] = 0 # we consider taking the action

            elif act == 2: # backward
                if pos + 1 < self.lane_length:
                    self.road[lane, pos + 1] = current_state[lane, pos]
                self.road[lane, pos] = 0 # we consider taking the action

            elif act == 3 and lane - 1 >= 0: # forward right
                if pos - 1 >= 0:
                    self.road[lane - 1, pos - 1] = current_state[lane, pos]
                self.road[lane, pos] = 0

            elif act == 4 and lane + 1 < self.num_lanes: # forward left
                if pos - 1 >= 0:
                    self.road[lane + 1, pos - 1] = current_state[lane, pos]
                self.road[lane, pos] = 0

        collision = self.check_collision(action, current_state)
        self.movement()
        self.target_slot -= 1

        done = self.check_done(collision)
        reward = self.get_reward(collision, done, (lane, pos), act, current_state)

        return self.window, reward, done, {"positions": self.positions}

    def check_collision(self, action, current_state):
        """
        This function checks if there is a collision
        """
        pos, act = self.encoding(action)
        lane = np.where(self.positions == pos)[0]
        pos = np.where(self.positions == pos)[1]

        if current_state[lane, pos] == 1:
            if act == 1 and pos - 1 >= 0 and\
                current_state[lane, pos-1] == current_state[lane, pos]:
                # collision
                return True

            elif act == 2 and pos + 1 < self.lane_length and\
                current_state[lane, pos+1] == current_state[lane, pos]:
                # collision
                return True

            elif act == 3 and lane - 1 >= 0 and pos-1>=0:
                if current_state[lane - 1, pos - 1] == current_state[lane, pos]:
                # collision
                    return True

            elif act == 4 and lane + 1 < self.num_lanes and pos-1 >=0:
                if current_state[lane + 1, pos - 1] == current_state[lane, pos]:
                # collision
                    return True

        return False

    def check_done(self, collision):
        """ checks if the episode has finished, the environment
        finishes when a free slot reaches the target lane at the
        merging point. """
        # we will assume that the merging point is position (2,1)
        if collision:
            return True
        else:
            if self.target_slot == 0:
                return True
            return False

    def reset(self):
        """ resets the environment """
        self.create_road()

        self.calculate_target_slot()
        return self.window

    def movement(self):
        """This function simulates the movement of the road at a certain speed"""
        # we move it to the left so it moves forward the lane as simulating the vehicules moving
        # forward all at the same speed
        self.road[:,:-1] = self.road[:,1:]
        new_column = np.random.binomial(n=1, size=(self.num_lanes, 1), p = 0.7)
        self.road[:,-1:] = new_column
        # if self.target_slot - 1 == self.lane_length-1:
        #     self.road[self.num_lanes-1, self.target_slot-1] = 1
        #     self.road[self.num_lanes-2, self.target_slot-1] = 1
        # if self.target_slot == self.lane_length-1:
        #         self.road[self.num_lanes-1, self.target_slot] = 1
        # if self.target_slot+1 == self.lane_length-1:
        #     self.road[self.num_lanes-1, self.target_slot+1] = 1

    def create_road(self):
        """
        This function creates the road with 3 lanes all with 0s representing the slots
        """
        prob = np.random.random()
        self.road = np.random.binomial(n=1, size=(self.num_lanes, self.lane_length), p = 0.7)
        self.window = self.road[:,:4]

    def render(self):
            """Render the road"""
            print("------------------------")
            for i in range(self.num_lanes):
                print(self.road[i])
                print("------------------------")