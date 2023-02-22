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
