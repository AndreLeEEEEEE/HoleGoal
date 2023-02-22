# This is the driver program

import gym
# The name of this module comes from setup.py
import HoleGoal
import numpy
from tqdm import tqdm, trange
import math
import os
from collections import deque
import time

# UI to display episode count
def print_episode(episode, delay=1):
    os.system('cls')
    for _ in range(17):
        print('=', end='')
    print("")
    print("Episode ", episode)
    for _ in range(17):
        print('=', end='')
    print("")
    time.sleep(delay)

# UI to display the world, delay of 1 sec for ease of understanding
def print_status(hg_env, action, done, step, delay=1, training_mode=True):
    os.system('cls')
    hg_env.print_world(action, step)
    if training_mode: hg_env.print_q_table()
    if done:
        print("-------EPISODE DONE--------")
        delay *= 2
    time.sleep(delay)

def main():
    verbose = True
    print('Do you want to train the agent? ',end=' ')
    ans = input('[y/N] ').lower()
    if ans in ['yes', 'y']:
        training_flag = True
    elif ans in ['no', 'n','N']:
        training_flag = False
    else:
        print("Didn't understand your input! Guessing you do not want to train at all!")
        training_flag = False

    if training_flag == True:
        maxwins = 100
        delay = 0
    else:
        maxwins = 5
        delay = 1

    wins = 0
    episode_count = 10 * maxwins
    # scores (max number of steps bef goal) - good indicator of learning
    scores = deque(maxlen=maxwins)

    # The name of this environment comes from HoleGoal/__init__.py
    hg_env = gym.make('hole-goal-v0', render_mode='human')

    if training_flag == False:
        #Load pre-existing Q-table
        print("Loading q-world...")
        hg_env.load_q_world()

    step = 1
    exit_flag = False
    # state, action, reward, next state iteration
    for episode in trange(episode_count):
        # At the start of an episode, reset the agent's position
        state = hg_env.reset()
        done = False
        print_episode(episode, delay=delay)
        while not done:
            print("Episode: ", episode + 1)
            # Use the policy to get the next action
            action = hg_env.act()
            # Apply the action to the current state
            packed_next_state, reward, done, truncated, info = hg_env.step(action)
            # The next_state had to be returned as a dictionary, but needs to be an int for later use
            next_state = packed_next_state["agent"]
            hg_env.update_q_table(state, action, reward, next_state)
            print_status(hg_env, action, done, step, delay=delay,training_mode=training_flag)
            hg_env.render()
            # Update state
            state = next_state
            # Check if the episode terminated
            if done:
                if hg_env.is_in_win_state():
                    wins += 1
                    scores.append(step)
                    if wins > maxwins:
                        exit_flag = True
                # Exploration-Exploitation is updated on episode end
                hg_env.update_epsilon()
                step = 1
            else:
                step += 1
            if exit_flag==True:
                break 
        if exit_flag == True:
            break
    if training_flag == True:
        print("Saving q-world for future use...")
        hg_env.save_q_world()
        print(scores)
        hg_env.print_q_table()

if __name__ == '__main__':
    main()
