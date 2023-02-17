import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class HoleGoalEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    # 4 actions
    # 0 - Left, 1 - Down, 2 - Right, 3 - Up
    self.col = 4
    self.action_space = (4,)

    # 16 states, 4x4 grid
    self.row = 16
    self.observation_space = (4,4)

    # setup the environment
    self.q_table = np.zeros([self.row, self.col])
    self.init_transition_table()
    self.init_reward_table()

    # discount factor
    self.gamma = 0.9

    # 90% exploration, 10% exploitation
    self.epsilon = 0.9
    # exploration decays by this factor every episode
    self.epsilon_decay = 0.9
    # in the long run, 10% exploration, 90% exploitation
    # meaning, the agent is never going to bore you with predictable moves :D
    self.epsilon_min = 0.1

    # reset the environment
    self.reset()
    self.is_explore = True

  def init_reward_table(self):
    """
    0 - Left, 1 - Down, 2 - Right, 3 - Up
--------------------------------------------
|         |          |          |          |
|  Start  |          |          |          |
|    0    |    1     |    2     |    3     |
--------------------------------------------
|         |          |  -100    |          |
|         |          |  Hole    |          |
|    4    |    5     |    6     |    7     |
--------------------------------------------
|         |  -100    |  +100    |          |
|         |  Hole    |  Goal    |          |
|    8    |    9     |    10    |    11    |
--------------------------------------------
|         |          |          |  -100    |
|         |          |          |  Hole    |
|   12    |   13     |    14    |    15    |
--------------------------------------------
    """
    self.reward_table = np.zeros([self.row, self.col])
    self.reward_table[5, 2] = -100.
    self.reward_table[5, 1] = -100.
    self.reward_table[2, 1] = -100.
    self.reward_table[7, 0] = -100.
    self.reward_table[8, 2] = -100.
    self.reward_table[11,0] = 100.
    self.reward_table[11,1] = -100.
    self.reward_table[13,3] = -100.
    self.reward_table[14,3] = 100.
    self.reward_table[14,2] = -100.


  def init_transition_table(self):
    """ TT[state_{i},action] = state_{i+1}
    0 - Left, 1 - Down, 2 - Right, 3 - Up
    ------------------
    | 0 | 1 | 2   | 3 |
    -------------------
    | 4 | 5 | 6H  | 7 |
    -------------------
    | 8 | 9H | 10G| 11| 
    -------------------
    |12 | 13 | 14 |15H|
    -------------------
    """
    left, down, right, up = (0,1,2,3)
    self.transition_table = np.zeros([self.row, self.col], dtype=int)

    T = np.zeros_like(self.transition_table)
    #non-goal and non-hole transitions
    T[0,left], T[0,right], T[0,up], T[0, down] = (0,1,0,4)
    T[1,left], T[1,right], T[1,up], T[1, down] = (0,2,1,5)
    T[2,left], T[2,right], T[2,up], T[2, down] = (1,3,2,6)
    T[3,left], T[3,right], T[3,up], T[3, down] = (2,3,3,7)
    T[4,left], T[4,right], T[4,up], T[4, down] = (4,5,0,8)
    T[5,left], T[5,right], T[5,up], T[5, down] = (4,6,1,9)
    T[7,left], T[7,right], T[7,up], T[7, down] = (6,7,3,11)
    T[8,left], T[8,right], T[8,up], T[8, down] = (8,9,4,12)
    T[11,left], T[11,right], T[11,up], T[11, down] = (10,11,7,15)
    T[12,left], T[12,right], T[12,up], T[12, down] = (12,13,8,12)
    T[13,left], T[13,right], T[13,up], T[13, down] = (12,14,9,13)
    T[14,left], T[14,right], T[14,up], T[14, down] = (13,15,10,14)

    # terminal Goal state
    T[10,left], T[10,right], T[10,up], T[10,down] = (10,10,10,10)

    # terminal Hole state
    T[6,left], T[6,right], T[6,up], T[6, down] = (6,6,6,6)
    T[9,left], T[9,right], T[9,up], T[9, down] = (9,9,9,9)
    T[15,left], T[15,right], T[15,up], T[15, down] = (15,15,15,15)

    self.transition_table = T

  def step(self, action):
    print('self.transition_table')
    print(self.transition_table)
    # determine the next_state given state and action
    next_state = self.transition_table[self.state, action]

    # done is True if next_state is Goal or Hole
    done = next_state == 6 or next_state == 9 or next_state == 10 or next_state == 15

    # reward given the state and action
    reward = self.reward_table[self.state, action]

    # the enviroment is now in new state
    self.state = next_state

    return next_state, reward, done

  # determine the next action
  def act(self):
    # 0 - Left, 1 - Down, 2 - Right, 3 - Up

    # action is from exploration, by following the distribution "epsilon"
    if np.random.rand() <= self.epsilon:
      # explore - do random action
      self.is_explore = True

      #find valid transitions from current state
      valid_actions_from_state = np.where(self.transition_table[self.state,:]!=self.state)
      #print('valid_actions_from_state {} = {}'.format(self.state,valid_actions_from_state[0]))
      #return np.random.choice(4, 1)[0]  #4C1
      return np.random.choice(valid_actions_from_state[0])

    # otherwise,  action is from exploitation
    # exploit - choose action with max Q-value
    self.is_explore = False

    return np.argmax(self.q_table[self.state])

  def reset(self):
    self.state = 0
    return self.state
    
  def render(self, mode='human', close=False):
    ...
  def close(self):
    ...