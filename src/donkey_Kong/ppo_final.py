# This code is without simulating climb up the ladder.




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

print(gym.__version__)

gym.register_envs(ale_py)

# Configuration file
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 20000000, 
    "Algo": "PPO",
    "env_name": "ALE/DonkeyKong-v5",
    "model_name": "DonkeyKong-v5",
    "Add": "ppo_final",
    "export_path": "../../models/donkey_kong/",
    "videos_path": "../../videos/donkey_kong/",
}

# from google.colab import userdata
# secret_value_0 = userdata.get('wandb')

# Wandb setup
wandb.login(key="YOUR_API_KEY")
run = wandb.init(
    project="Project_1",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    save_code=True,  # optional
)

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

        # Update the observation space to reflect the new transposed shape
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
        Transposes observations to (C, H, W) format if `skip` is False.

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


# This code is made with help of chatgpt
def get_agent_level_position(image):
    # This function processes an input grayscale image to:
    # 1. Detect horizontal lines to define levels.
    # 2. Identify the position of an agent.
    # 3. Determine the level the agent is located on based on detected lines.
    # Input: 
    #   - image: A grayscale image (NumPy array).
    # Output: 
    #   - agent_level: Integer representing the agent's level (1-based index).
    #   - agent_position: Tuple (x, y) representing the agent's position in the image.
    #
    # Raises:
    #   - ValueError: If the input image is None.

    if image is None:
        raise ValueError("Image not loaded. Check the path and file.")

    # remove the top 32 pixels
    image = image[32:, :]

    # Zero out a specific region of the image (mask irrelevant parts)
    image[149:160, 36:44] = 0

    # Lines detection
    # copy image
    gray_image = image.copy()


    # Perform edge detection
    edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)

 

    # Detect horizontal lines using Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=10,
        maxLineGap=20
    )

    # Draw detected lines on a debug image
    debug_line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    horizontal_lines = []

    if lines is not None:
        # print(f"Total lines detected (before filtering): {len(lines)}")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check for horizontal lines with a more lenient threshold
            vertical_diff = abs(y2 - y1)
            horizontal_diff = abs(x2 - x1)

            if vertical_diff < horizontal_diff * 0.1:  # Allow slight vertical tilt
                horizontal_lines.append((x1, y1, x2, y2))


    # detect the agent and it position
    # Perform binary thresholding to highlight the agent and objects
    _, binary = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


    # Detect contours in the cleaned binary image
    contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for the agent's position
    agent_detection_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    agent_position = None

    # Filter contours to find the agent

    for contour in contours:
        # Get bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter based on size: Assuming agent is small
        if 5 <= w <= 30 and 5 <= h <= 30:  # Adjust range based on resolution
            # Further filter based on aspect ratio to avoid line-like objects
            aspect_ratio = max(w / h, h / w)
            if aspect_ratio < 2.0:  # Allow only nearly square contours
                agent_position = (x + w // 2, y + h // 2)  # Center of the bounding box

                break  # Assuming only one agent in the frame

    # Detect level
    # Sort lines by their average y-value, descending (bottom to top)
    lines_sorted = sorted(horizontal_lines, key=lambda line: (line[1] + line[3]) / 2, reverse=True)

    # Function to group horizontal lines into clusters based on proximity
    def cluster_lines(lines, desired_clusters=7, proximity_threshold=10):
        clusters = []
        current_cluster = [lines[0]]
        for line in lines[1:]:
            line_y = (line[1] + line[3]) // 2
            current_cluster_y = sum((l[1]+l[3])//2 for l in current_cluster) / len(current_cluster)
            # If the difference is small, add to current cluster, else start a new one
            if abs(line_y - current_cluster_y) < proximity_threshold:
                current_cluster.append(line)
            else:
                clusters.append(current_cluster)
                current_cluster = [line]
        clusters.append(current_cluster)

        return clusters

    proximity_threshold = 10  # Adjust as needed
    clusters = cluster_lines(lines_sorted, desired_clusters=7, proximity_threshold=proximity_threshold)

    # Compute representative y-value for each cluster (average)
    boundary_y_values = []
    for cluster in clusters:
        avg_y = sum((l[1] + l[3]) // 2 for l in cluster) / len(cluster)
        boundary_y_values.append(avg_y)

    # Sort boundaries again in descending order (bottom = largest y, top = smallest y)
    boundary_y_values.sort(reverse=True)

    agent_level = None
    if agent_position:
        agent_y = agent_position[1]
        # Find which level agent_y falls into
        for i in range(6):
            if boundary_y_values[i] >= agent_y > boundary_y_values[i+1]:
                agent_level = i + 1
                break


    # Draw minimal annotation: just draw the agent and print its level
    final_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


    return agent_level, agent_position# agent position is (x, y)

# This code is made with help of chatgpt
class IntermediateRewardWrapper(gym.Wrapper):
# Input:
# - env: The base environment to wrap.

# Output:
# - The modified reward, which includes additional rewards for achieving specific levels,
#   reaching certain positions, and making progress toward the goal.
    def __init__(self, env):
        """
        Initializes the wrapper, setting up ladder positions and tracking variables.
        
        Parameters:
        - env: The environment being wrapped.
        """
        super(IntermediateRewardWrapper, self).__init__(env)
        self.ladder_postion = [110,82,90,70,110,78]
        self.last_level = 1
        self.previous_additional_reward = 0.0
        self.last_y_position = 0

    def step(self, action):
        """
        Executes an environment step, modifies the reward, and returns the results.

        Parameters:
        - action: The action taken by the agent.

        Returns:
        - obs: The new observation after the step.
        - reward: The modified reward, including the additional intermediate rewards.
        - terminated: Whether the episode is terminated.
        - truncated: Whether the episode is truncated.
        - info: Additional information from the environment.
        """

        obs, reward, terminated, truncated, info = self.env.step(action)

        additional_reward = 0.0

        # get agent level and position
        agent_level, agent_position = get_agent_level_position(obs)

 

        if agent_level is not None and 1 <= agent_level <= len(self.ladder_postion):
            agent_level_reward = (7 - agent_level)* (-0.01)
            additional_reward += agent_level_reward

            if agent_level > self.last_level:
                additional_reward += 5000.0
                self.last_level = agent_level
            

            diff = 0
            if agent_position is not None and len(agent_position) > 0:
                # get absolute difference between agent position and ladder position
                diff = abs(self.ladder_postion[agent_level - 1] - agent_position[0])
                agent_position_reward = diff * (-0.1)
                additional_reward += agent_position_reward

                if action == 2 and diff <= 1 and agent_position[1] > self.last_y_position:
                    additional_reward += 30.0
                self.last_y_position = agent_position[1]



        else:
            # If agent_level or agent_position not found, use the previous reward
            additional_reward = self.previous_additional_reward


        # Round the additional reward to 2 decimal places
        additional_reward = round(additional_reward, 2)

        # Update the previous reward
        self.previous_additional_reward = additional_reward

        # Add the additional reward to the original reward
        reward += additional_reward

        return obs, reward, terminated, truncated, info

class ActionFilterWrapper(gym.ActionWrapper):
# Input:
# - env: The environment to wrap.
# - allowed_actions: A list of allowed actions (subset of the original action space).

# Output:
# - A modified environment where the action space is reduced to only include `allowed_actions`.
    def __init__(self, env, allowed_actions):
        """
        Initializes the wrapper and modifies the action space.

        Parameters:
        - env: The base Gym environment to wrap.
        - allowed_actions: A list of allowed actions (subset of the original action space).
        """
        super().__init__(env)
        self.allowed_actions = allowed_actions
        # The new action space matches the number of allowed actions
        self.action_space = Discrete(len(self.allowed_actions))

    def action(self, act):
        """
        Maps an action index from the reduced action space back to the original action space.

        Parameters:
        - act: An integer representing the action index in the reduced action space.

        Returns:
        - The corresponding action from the original environment's action space.
        """
        return self.allowed_actions[act]

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



def make_env(env_name, allowed_actions, obs_type="grayscale", render_mode=None,):
# Input:
# - env_name: The name of the Gym environment (e.g., "ALE/Breakout-v5").
# - allowed_actions: A list of actions that are allowed in the environment.
# - obs_type: The type of observation, typically "grayscale" or other formats (default is "grayscale").
# - render_mode: An optional argument to specify the rendering mode (e.g., None, "rgb_array").
#
# Output:
# - A function (_init) that initializes and returns a wrapped Gym environment with multiple wrappers.

    def _init():
        env = gym.make(env_name, obs_type="grayscale", render_mode=render_mode)
        print("Standard Env.        : {}".format(env.observation_space.shape))
        env = FireResetEnv(env)
        print("FireResetEnv          : {}".format(env.observation_space.shape))
        # Wrap the environment with the custom ActionFilterWrapper
        env = ActionFilterWrapper(env, allowed_actions)
        print("ActionFilterWrapper   : {}".format(env.observation_space.shape))
        # Wrap the environment to add intermediate rewards
        env = IntermediateRewardWrapper(env)
        print("IntermediateReward    : {}".format(env.observation_space.shape))
        env = ResizeObservation(env, (84, 84))
        print("ResizeObservation    : {}".format(env.observation_space.shape))
        env = AddChannelDimension(env)  # Add channel dimension here
        print("AddChannelDimension  : {}".format(env.observation_space.shape))

        env = ScaledFloatFrame(env)
        print("ScaledFloatFrame     : {}".format(env.observation_space.shape))

        return env
    return _init


# select relevant actions
allowed_actions = [0, 1, 2, 3, 4, 11, 12]
env = make_vec_env(env_id=make_env(config["env_name"], allowed_actions= allowed_actions), n_envs=8)

# stack 4 frames
env = VecFrameStack(env, n_stack=4)
print("Post VecFrameStack Shape: {}".format(env.observation_space.shape))

# convert back to PyTorch format (channel-first)
env = MyVecTransposeImage(env)
print("VecTransposeImage Shape: {}".format(env.observation_space.shape))

print("Final Observation Space: {}".format(env.observation_space.shape))

print("Check")
print("Post VecFrameStack Shape: {}".format(env.observation_space.shape))
print("Final Observation Space: {}".format(env.observation_space.shape))

# Create an evaluation environment (similar to our training env)
eval_env = make_vec_env(env_id=make_env(config["env_name"], allowed_actions= allowed_actions), n_envs=1)

# stack 4 frames
eval_env = VecFrameStack(eval_env, n_stack=4)
print("eval_env Post VecFrameStack Shape: {}".format(eval_env.observation_space.shape))

# convert back to PyTorch format (channel-first)
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
            n_steps=2048,
            n_epochs=10,
            clip_range=0.1,
            vf_coef=0.5,
            ent_coef=0.01, # 0.01
            policy_kwargs=dict(net_arch=[512, 256], normalize_images=False),
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

# This code is made with help of chatgpt
class DebugObservationCallback(BaseCallback):
# Input:
# - verbose: Integer indicating verbosity level for logging (default is 0, meaning no verbose output).
#
# Output:
# - Prints debug information about the observations (mean, standard deviation, min, and max) at each training step.

    def __init__(self, verbose=0):
        """
        Initializes the callback.

        Parameters:
        - verbose: The verbosity level (default is 0).
        """
        super(DebugObservationCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method is called after every training step.

        It prints the mean, standard deviation, minimum, and maximum values of the current observation.

        Returns:
        - True to continue training after the step.
        """
        # Get the current observation
        observation = self.locals["new_obs"]  # Observations from the environment
        print("Observation mean:", np.mean(observation))
        print("Observation std:", np.std(observation))
        print("Observation min:", np.min(observation))
        print("Observation max:", np.max(observation))


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