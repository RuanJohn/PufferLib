import functools
import time

import gymnasium as gym
import numpy as np
import rware  # noqa: F401
from gymnasium import spaces
from pettingzoo import ParallelEnv

import pufferlib
import pufferlib.vector

NUM_ENVS = 64
NUM_WORKERS = 4
ENV_BATCH_SIZE = NUM_ENVS
VEC_ZERO_COPY = True
BACKEND = pufferlib.vector.Multiprocessing
ENV_NAME = "rware-tiny-2ag-v2"


class PettingZooRWARE(ParallelEnv):
    """
    PettingZoo Parallel API wrapper for RWARE environments.

    Example usage:
        env = gym.make("rware-tiny-2ag-v2", sensor_range=3, request_queue_size=6)
        env = PettingZooRWARE(env)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "rware_v0",
        "is_parallelizable": True,
        "render_fps": 10,
    }

    def __init__(self, env):
        """
        Initialize the PettingZoo wrapper.

        Args:
            env: A gymnasium-wrapped RWARE environment
        """
        self.env = env

        # PettingZoo specific attributes
        self.possible_agent_names = [f"agent_{i}" for i in range(self.env.n_agents)]
        self.agents = self.possible_agent_names.copy()
        self.possible_agents = self.possible_agent_names.copy()

        # Create action and observation spaces for each agent
        self.action_spaces = {
            agent: self._convert_to_pettingzoo_space(self.env.action_space[i])
            for i, agent in enumerate(self.possible_agent_names)
        }

        self.observation_spaces = {
            agent: self._convert_to_pettingzoo_space(self.env.observation_space[i])
            for i, agent in enumerate(self.possible_agent_names)
        }

    def _convert_to_pettingzoo_space(self, space):
        """Convert Gymnasium space to PettingZoo compatible space."""
        if isinstance(space, spaces.Tuple):
            if len(space.spaces) == 1:
                return space.spaces[0]
            return spaces.Tuple(space.spaces)
        return space

    def reset(self, seed=None, options=None):
        """Reset the environment and return observations for each agent."""
        self.agents = self.possible_agent_names.copy()
        observations, _ = self.env.reset(seed=seed, options=options)

        return {agent: obs for agent, obs in zip(self.agents, observations)}, {}

    def step(self, actions):
        """
        Step the environment with actions from all agents.

        Args:
            actions: Dict mapping agent names to their actions

        Returns:
            observations: Dict mapping agent names to their observations
            rewards: Dict mapping agent names to their rewards
            terminations: Dict mapping agent names to their termination status
            truncations: Dict mapping agent names to their truncation status
            infos: Dict mapping agent names to their info dictionaries
        """
        # Convert dict of actions to list in correct agent order
        action_list = [actions[agent] for agent in self.agents]

        observations, rewards, done, truncated, infos = self.env.step(action_list)

        # Convert to PettingZoo format (dict with agent names as keys)
        observations_dict = {
            agent: obs for agent, obs in zip(self.agents, observations)
        }

        rewards_dict = {agent: reward for agent, reward in zip(self.agents, rewards)}

        # In PettingZoo, we need termination status per agent
        terminations = {agent: done for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        # Expand info dict for each agent
        infos = {agent: infos.copy() for agent in self.agents}

        return observations_dict, rewards_dict, terminations, truncations, infos

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    def state(self) -> np.ndarray:
        """
        Return the global state of the environment.
        Uses the built-in global image observation from the warehouse environment.
        """
        return self.env.get_global_image()

    @property
    def unwrapped(self):
        """Return the base environment."""
        return self.env


def env_creator(name="rware-tiny-2ag-v2"):
    return functools.partial(make, name)


def make(name):
    with pufferlib.utils.Suppress():
        env = gym.make(name)
    env = PettingZooRWARE(env)
    # env = pufferlib.postprocess.MultiagentEpisodeStats(env)
    # env = pufferlib.postprocess.MeanOverAgents(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env)


def main():
    make_env = env_creator(ENV_NAME)

    vecenv = pufferlib.vector.make(
        make_env,
        num_envs=NUM_ENVS,
        num_workers=NUM_WORKERS,
        batch_size=ENV_BATCH_SIZE,
        zero_copy=VEC_ZERO_COPY,
        backend=BACKEND,
    )

    done = False
    observations, _ = vecenv.reset()
    steps = 0
    start_time = time.time()
    while not done:
        actions = vecenv.action_space.sample()
        observations, rewards, terminations, truncations, infos = vecenv.step(actions)
        done = all(terminations + truncations)
        steps += NUM_ENVS

    print(f"Steps per second: {steps / (time.time() - start_time)}")
    vecenv.close()


if __name__ == "__main__":
    main()
