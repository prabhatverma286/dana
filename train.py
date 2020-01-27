import json
import os
import sys

import gym
from gym import wrappers, logger
import argparse

from agent import DQNAgent, save_video_and_stats


def main(env_id, identifier, arguments):
    logger.set_level(logger.INFO)

    env = gym.make(env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = 'agent-training'
    env = wrappers.Monitor(env, directory=outdir, force=True, video_callable=save_video_and_stats)
    env.seed(0)
    agent = DQNAgent(outdir, env, identifier)

    agent.train(env, arguments)
    agent.save("saved_models\\" + env_id + "_model.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', help='gym env id')
    parser.add_argument('--identifier', default=None, help='gym env id')
    parser.add_argument('--json_arguments', type=json.loads, help='gym env id')

    args = parser.parse_args()
    main(args.env_id, args.identifier, args.json_arguments)
