import os

import gym
from gym import wrappers, logger
import argparse

from agent import DQNAgent


def save_video_and_stats(episode_number):
    return episode_number % 25 == 0 and episode_number is not 0


def main(env_id):
    logger.set_level(logger.INFO)

    env = gym.make(env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = os.path.join('random-agent-results', env_id)
    env = wrappers.Monitor(env, directory=outdir, force=True, video_callable=save_video_and_stats)
    env.seed(0)
    agent = DQNAgent(outdir, env)

    agent.train()
    agent.save("saved_models\\" + env_id + "_model.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', help='gym env id')
    args = parser.parse_args()
    main(args.env_id)
