import numpy as np
import dill
import os

import agent
import environment

# Set obstacles position
obstacle= [[0, 0], [0, 7],
           [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 7],
           [3, 4],
           [4, 4],
           [5, 4],
           [6, 4],
           [7, 4],
           [8, 4],
           [9, 9]]

# Set sand position
sand = [[2, 7],
        [3, 3], [3, 5], [3, 6], [3, 7],
        [5, 0], [5, 1], [5, 2]]

# Setting Goal
goal = [0, 3]

# Setting training lenghts
episodes = 1500         # number of training episodes
episode_length = 50     # maximum episode length

# Setting matrix dimension
x = 10                  # horizontal size of the box
y = 10                  # vertical size of the box

# Agent parameters
discount = 0.9          # exponential discount factor
softmax = True          # set to true to use Softmax policy
sarsa = True            # set to true to use the Sarsa algorithm

# Others params
alpha = np.ones(episodes) * 0.5
epsilon = np.linspace(0.1, 0.001,episodes)
sand_penalization = -1

# Initialize the agent
learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)

print('\nStarting {} episodes of training with SARSA...'.format(episodes))
rewards_list = []
position_list = []
# Perform the training
for index in range(0, episodes):
    # Start from a random state
    initial = [np.random.randint(0, x), np.random.randint(0, y)]
    position_list.append(initial)
    # Initialize environment
    state = initial
    env = environment.Environment(x, y, state, goal, sand_penalization=sand_penalization)
    env.create_obstacle(obstacle)
    env.create_sand(sand)
    reward = 0
    # run episode
    for step in range(0, episode_length):
        # Find state index
        state_index = state[0] * y + state[1]
        # Choose an action
        action = learner.select_action(state_index, epsilon[index])
        # the agent moves in the environment
        result = env.move(action)
        # Update learner
        next_index = result[0][0] * y + result[0][1]
        learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
        # Update state and reward
        reward += result[1]
        state = result[0]
    # Save result
    reward /= episode_length
    rewards_list.append(reward)
    if (index)%100 == 0:
        print('Episode ', index, ': the agent has obtained an average reward of ', np.mean(rewards_list[-100:]))
print('Episode ', episodes, ': the agent has obtained an average reward of ', np.mean(rewards_list[-100:]))


# Saving Agent
print('\nSaving agent at trained_model\\learner.obj')
# Create dir if it doesn't exist
if os.path.isdir('trained_model') == False:
    os.mkdir('trained_model')
with open('trained_model\\learner.obj', 'wb') as agent_file:
    dill.dump(agent, agent_file)

# Saving results
print('Saving log file at trained_model\\log.txt')
with open('trained_model\\log.txt', 'w') as log_file:
    for index in range(len(rewards_list)):
        str = 'Episode {}: the agent has obtained an average reward of {} starting from position {}\n'.format(index+1, rewards_list[index], position_list[index])
        log_file.write(str)
