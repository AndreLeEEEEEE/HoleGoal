# HoleGoal

## How to set up the environment

1. Set up and activate the virtual environment (commands below).

- On Windows and Command Prompt: `$ env\Scripts\activate.bat`

- On Mac: `$ source env/bin/activate`

2. Use `$ pip install -e .` or `$ pip install -r requirements.txt` for the modules.

## How to run the driver file: [my_gym_demo.py](./my_gym_demo.py)

1. If on Windows, type `$ python my_gym_demo.py`. If on Mac and the driver file has been turned into an executable with `$ chmod +x my_gym_demo.py`, type `$ ./my_gym_demo.py`. If the executable method doesn't work, just use python interpreter to run the driver file.

2. (Optional since the [trained data](./joblibs/a_q_world_v1.joblib) is in this repo) Run the driver file in training mode

- If training automatically stops after ~300-400 episodes, the model should be trained correctly. If training persists longer than ~500 episodes, the agent will most likely get itself stuck on the left side of the environment during the actual execution cycle; at this point, the agent will almost never reach the goal. If this happens restart training.

3. Run the driver file in non-training mode.

## The process

Most of the logic for the custom HoleGoal environment is in [hole_goal_env.py](./HoleGoal/envs/hole_goal_env.py). Within this file is the HoleGoalEnv class, which inherits from gym.Env. The __init__() function builds the action space, observation space, discount factor, epsilon factors, q table, transitions table, and rewards table. Other functions required by gym such as action(), step(), and reset() are contained here. Last, most functions for printing the q table, transitions table, and environment are here.

The driver file first starts by asking if the user wants to conduct training. If the answer is no, the driver loads in a pre-existing q table and runs the reinforcement learning environment for 50 episodes. If the answer is yes, a pre-existing q table isn't loaded in and training begins. Training runs until 1000 episodes have terminated, but it can stop sooner. The q table generated at the end of training is saved for later use.

Regardless of which option the user chooses, this is the general procedure the driver executes. 

1. Load in the custom HoleGoal environment.

2. At the start of an episode, call the environment's reset() function.

3. Call the environment's act() function to find a valid action to take from the agent's current position. Generate a random number and compare it to epsilon. If less than epsilon, enter the explore state and pick a random action. If more than epsilon, pick the valid action with the largest q value. 

4. Call the environment's step() function to apply the action to the current state and get the next state. The transitions table and rewards tables are consulted and updated respectively.

5. Call the environment's update_q_table() function to update the q table using the current state, next state, reward, and action.

6. Print the world as it is using the driver's print_status() function. This, in turn, also calls printing functions in the environment.

7. Update the current state.

8. Check if the agent is now in a terminating state. If yes, call the environment's update_epsilon() function to update epsilon with the epsilon decay value. If no, the  episode ends anyway.

9. If there are still episodes to be run, go back to step 2. If not, end the reinforcement learning cycle. 
