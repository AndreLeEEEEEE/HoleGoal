from gym.envs.registration import register
# The last part of the entry point is the environment class from HoleGoal/envs/hole_goal_env.py
register(
    id='hole-goal-v0',
    entry_point='HoleGoal.envs:HoleGoalEnv',
)