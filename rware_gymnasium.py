import time

import gymnasium as gym
import rware  # noqa: F401

ENV_NAME = "rware-tiny-2ag-v2"
NUM_ENVS = 64


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        return env

    return thunk


if __name__ == "__main__":
    envs = gym.vector.AsyncVectorEnv(
        [make_env(ENV_NAME) for i in range(NUM_ENVS)],
    )

    obs, _ = envs.reset()
    done = False
    steps = 0
    start_time = time.time()

    while not done:
        actions = envs.action_space.sample()
        obs, rewards, terminations, truncations, infos = envs.step(actions)
        done = all(terminations + truncations)
        steps += NUM_ENVS

    print(f"Steps per second: {steps / (time.time() - start_time)}")
    envs.close()
