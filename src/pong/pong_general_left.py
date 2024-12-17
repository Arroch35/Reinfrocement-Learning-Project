

import warnings
warnings.filterwarnings('ignore')
import ale_py
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import torch
import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

from gymnasium.wrappers import ResizeObservation
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import collections
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper
import cv2
from gymnasium.spaces import Discrete
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
import shutil

print(gym.__version__)

gym.register_envs(ale_py)

# configuration file
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 2000000, # 1000000, 3000000, 20000000
    "Algo": "PPO",
    "env_name": "PongNoFrameskip-v4",
    "model_name": "PongNoFrameskip-v4",
    "Add": "left",
    "export_path": "../../models/pong/",
    "videos_path": "../../videos/pong/",
}


# Wandb setup
wandb.login(key="Your_API_Key")
run = wandb.init(
    project="pong",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    save_code=True,  # optional
)

# This code is made with help of chatgpt
class MyVecTransposeImage(VecEnvWrapper):
# Input:
# - venv: A vectorized environment (e.g., DummyVecEnv or SubprocVecEnv).
# - skip: A boolean flag to disable transposing if set to True (default is False).
#
# Output:
# - Transposed image observations in the format (Channels, Height, Width).
# - If skip is True, the observations remain unchanged.
    def __init__(self, venv, skip=False):
        """
        Initializes the wrapper and modifies the observation space to reflect the new shape.

        Parameters:
        - venv: The vectorized environment being wrapped.
        - skip: Flag to skip transposing observations. Default is False.
        """ 
        super().__init__(venv)
        self.skip = skip

        # Get original shape: e.g., (84, 84, 4)
        old_shape = self.observation_space.shape
        # Transpose shape to (C, H, W)
        new_shape = (old_shape[2], old_shape[0], old_shape[1])  # (4, 84, 84)

        # Use the original low/high if they are uniform; if not, use min/max appropriately
        low_val = self.observation_space.low.min()
        high_val = self.observation_space.high.max()

        self.observation_space = gym.spaces.Box(
            low=low_val,
            high=high_val,
            shape=new_shape,
            dtype=self.observation_space.dtype
        )

    def reset(self):
        """
        Resets the environment and returns the initial observation after transposing.
        
        Returns:
        - Transposed observation (or original if `skip=True`).
        """
        obs = self.venv.reset()
        return self.transpose_observations(obs)

    def step_async(self, actions):
        """
        Sends actions to the environment asynchronously.
        
        Parameters:
        - actions: Actions to be taken in the environment.
        """
        self.venv.step_async(actions)

    def step_wait(self):
        """
        Waits for the step to complete and processes the results.

        Returns:
        - Transposed observations, rewards, dones, and infos.
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        return self.transpose_observations(obs), rewards, dones, infos

    def transpose_observations(self, obs):
        """
        Transposes observations to (C, H, W) format if skip is False.

        Parameters:
        - obs: Observations from the environment.

        Returns:
        - Transposed observations in the desired shape.
        """
        if self.skip:
            return obs
        if isinstance(obs, dict):
            for key, val in obs.items():
                obs[key] = self._transpose(val)
            return obs
        else:
            return self._transpose(obs)

    def _transpose(self, obs):
        # obs shape is (n_envs, H, W, C) -> transpose to (n_envs, C, H, W)
        return obs.transpose(0, 3, 1, 2)



class ScaledFloatFrame(gym.ObservationWrapper):
# Input:
# - env: The base environment to wrap.

# Output:
# - Observations are returned as floats in the range [0.0, 1.0] instead of integers [0, 255].
    def __init__(self, env):
        """
        Initializes the wrapper and updates the observation space.

        Parameters:
        - env: The base Gym environment to wrap.
        """
        super().__init__(env)
        # The original shape remains (84,84,1), but the dtype and range change
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, obs):
        """
        Scales the observation pixel values from [0, 255] to [0, 1].

        Parameters:
        - obs: The original observation (e.g., an image with pixel values 0-255).

        Returns:
        - Scaled observation as a float32 array with values in [0.0, 1.0].
        """
        return np.array(obs).astype(np.float32) / 255.0




class InvertPongWrapper(gym.ObservationWrapper):
    """
    Wrapper to invert the observations and actions to train the left paddle in Pong.
    """
    def __init__(self, env):
        super().__init__(env)

        # Directly copy the low and high values from the original observation space
        obs_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            low=obs_space.low,  # No need to flip as values are uniform
            high=obs_space.high,
            shape=obs_space.shape,
            dtype=obs_space.dtype,
        )

    def observation(self, obs):
        """
        Flip the screen horizontally so that the left paddle is treated as the primary agent.
        """
        return np.flip(obs, axis=1)  # Flip the width axis

    def step(self, action):
        """
        Flip the actions to control the left paddle.
        """
        # Invert the action logic
        if action == 2:  # RIGHT
            action = 3  # LEFT
        elif action == 3:  # LEFT
            action = 2  # RIGHT
        elif action == 4:  # RIGHTFIRE
            action = 5  # LEFTFIRE
        elif action == 5:  # LEFTFIRE
            action = 4  # RIGHTFIRE

        # Perform the step with the flipped action
        return super().step(action)



class FireResetEnv(gym.Wrapper):
# Input:
# - env: The base environment to wrap.

# Output:
# - The environment resets normally, but performs the "FIRE" action to start the game.
    def __init__(self, env=None):
        """
        Initializes the wrapper and ensures the environment supports the 'FIRE' action.

        Parameters:
        - env: The base Gym environment to wrap.
        """
        super().__init__(env)
        # Check that 'FIRE' is a valid action in the environment
        assert 'FIRE' in env.unwrapped.get_action_meanings(), "Environment does not support 'FIRE' action"
        assert len(env.unwrapped.get_action_meanings()) >= 3, "Action space too small for expected actions"

    def step(self, action):
        """
        Passes the action to the environment's step function.

        Parameters:
        - action: The action to execute.

        Returns:
        - Tuple containing (observation, reward, terminated, truncated, info).
        """
        return self.env.step(action)

    def reset(self, **kwargs):
        """
        Resets the environment and performs the 'FIRE' action to start the game.

        Parameters:
        - kwargs: Additional arguments for the reset function.

        Returns:
        - obs: The initial observation after performing the 'FIRE' action.
        - info: Additional environment information.
        """
        # Reset the environment
        obs, info = self.env.reset(**kwargs)

        # Perform the FIRE action
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:  # If game ends after FIRE, reset again
            obs, info = self.env.reset(**kwargs)

        return obs, info

# Custom wrapper to add channel dimension
class AddChannelDimension(gym.ObservationWrapper):
# Input:
# - env: The base environment to wrap.

# Output:
# - Observations are returned with an additional channel dimension, e.g., (H, W) -> (H, W, 1).
    def __init__(self, env):
        """
        Initializes the wrapper and updates the observation space to include a channel dimension.

        Parameters:
        - env: The base Gym environment to wrap.
        """
        super().__init__(env)
        obs_shape = self.observation_space.shape
        # Update the observation space to include a channel dimension
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(obs_shape[0], obs_shape[1], 1),
            dtype=np.uint8,
        )

    def observation(self, observation):
        """
        Adds a channel dimension to the observation.

        Parameters:
        - observation: The original observation (e.g., shape (H, W)).

        Returns:
        - Modified observation with shape (H, W, 1).
        """
        # Add a channel dimension
        return np.expand_dims(observation, axis=-1)



def make_env(env_name, obs_type="grayscale", render_mode=None, invert=False):
# Input:
# - env_name: The name of the Gym environment (e.g., "ALE/Breakout-v5").
# - allowed_actions: A list of actions that are allowed in the environment.
# - obs_type: The type of observation, typically "grayscale" or other formats (default is "grayscale").
# - render_mode: An optional argument to specify the rendering mode (e.g., None, "rgb_array").
# If `invert=True`, the environment will be flipped for the left paddle
#
# Output:
# - A function (_init) that initializes and returns a wrapped Gym environment with multiple wrappers.

    def _init():
        env = gym.make(env_name, obs_type="grayscale", render_mode=render_mode)
        print("Standard Env.        : {}".format(env.observation_space.shape))
        env = FireResetEnv(env)
        print("FireResetEnv          : {}".format(env.observation_space.shape))
        env = ResizeObservation(env, (84, 84))
        print("ResizeObservation    : {}".format(env.observation_space.shape))
        env = AddChannelDimension(env)
        print("AddChannelDimension  : {}".format(env.observation_space.shape))
        env = ScaledFloatFrame(env)
        print("ScaledFloatFrame     : {}".format(env.observation_space.shape))

        # Invert the environment for left paddle training
        if invert:
            env = InvertPongWrapper(env)
            print("InvertPongWrapper Applied : {}".format(env.observation_space.shape))

        return env
    return _init


# Create environment for the left paddle
env = make_vec_env(env_id=make_env(config["env_name"], invert=True), n_envs=8)

# stack 4 frames
env = VecFrameStack(env, n_stack=4)
print("Post VecFrameStack Shape: {}".format(env.observation_space.shape))

# convert back to PyTorch format (channel-first)
# env = VecTransposeImage(env)
env = MyVecTransposeImage(env)
print("VecTransposeImage Shape: {}".format(env.observation_space.shape))

print("Final Observation Space: {}".format(env.observation_space.shape))

print("Check")
print("Post VecFrameStack Shape: {}".format(env.observation_space.shape))
print("Final Observation Space: {}".format(env.observation_space.shape))

# Create the evaluation environment for the left paddle
eval_env = make_vec_env(env_id=make_env(config["env_name"], invert=True), n_envs=8)

# stack 4 frames
eval_env = VecFrameStack(eval_env, n_stack=4)
print("eval_env Post VecFrameStack Shape: {}".format(eval_env.observation_space.shape))

# convert back to PyTorch format (channel-first)
# eval_env = VecTransposeImage(eval_env)
eval_env = MyVecTransposeImage(eval_env)
print("eval_env MyVecTransposeImage Shape: {}".format(eval_env.observation_space.shape))
print("eval_env Final Observation Space: {}".format(eval_env.observation_space.shape))

model = PPO(config["policy_type"],
            env,
            verbose=0,
            tensorboard_log=f"runs/{run.id}",
            batch_size=256,
            learning_rate=2.5e-4, #2.5e-4, 0.001
            gamma=0.99,
            n_steps=128,
            n_epochs=4,
            clip_range=0.1,
            vf_coef=0.5,
            ent_coef=0.01, # 0.01
            policy_kwargs=dict(normalize_images=False),
            device="cuda")


# Create the evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=config["export_path"],  # directory to save the best model
    log_path=config["export_path"],              # evaluation logs
    eval_freq=150000,                            # evaluate the model every 150,000 steps
    deterministic=True,
    render=False
)

# This code is made with help of chatgpt
class GradientInspectionCallback(BaseCallback):
# Input:
# - verbose: Integer indicating the verbosity level for logging (default is 0, meaning no verbose output).
#
# Output:
# - Logs the gradient norms of each parameter in the policy network during training steps.
    def __init__(self, verbose=0):
        """
        Initializes the callback.

        Parameters:
        - verbose: The verbosity level (default is 0).
        """
        super(GradientInspectionCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method is called after every training step.

        It inspects the gradients of the policy network and logs the norm of each gradient.

        Returns:
        - True to continue training after the step.
        """
        # Access the policy network
        policy_net = self.model.policy

        # Iterate over the parameters to inspect gradients
        for name, param in policy_net.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.logger.record(f"gradients/{name}_norm", grad_norm)

        return True  # Continue training


# Combine both callbacks
# callbacks = CallbackList([eval_callback, WandbCallback(verbose=2), GradientInspectionCallback(), DebugObservationCallback()])
callbacks = CallbackList([eval_callback, WandbCallback(verbose=2), GradientInspectionCallback()])

# train
t0 = datetime.now()
model.learn(total_timesteps=config["total_timesteps"], callback=callbacks)
t1 = datetime.now()
print('>>> Training time (hh:mm:ss.ms): {}'.format(t1-t0))

# save and export model
model.save(config["export_path"] + config["model_name"])

"""Training time without parallezing env: >>> Training time (hh:mm:ss.ms): 1:01:58.577332"""

# finish wandb project
wandb.finish()

print("Finish")