import json
import os
import sys

import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from gym import wrappers, logger
import argparse

from gym.wrappers import Monitor

from Helpers import envs, BreakoutMonitor
from agent import DQNAgent, save_video_and_stats


def main(env_id, identifier, arguments):
    logger.set_level(logger.INFO)

    if env_id == envs["Cartpole"]:
        env = gym.make(env_id)
    else:
        env = make_atari(env_id)
        env = wrap_deepmind(env, frame_stack=True, scale=False, clip_rewards=False)

    out_dir = 'agent-training'
    os.makedirs(out_dir, exist_ok=True)

    if env_id == envs["Breakout"]:
        env = BreakoutMonitor(env, directory=out_dir, force=True, video_callable=save_video_and_stats)
    else:
        env = Monitor(env, directory=out_dir, force=True, video_callable=save_video_and_stats)

    env.seed(0)
    agent = DQNAgent(out_dir, env, identifier)

    agent.train(env, arguments)
    agent.save("saved_models\\" + env_id + "_model.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', help='gym env id')
    parser.add_argument('--identifier', default=None, help='gym env id')
    parser.add_argument('--json_arguments', type=json.loads, help='gym env id')

    args = parser.parse_args()
    main(args.env_id, args.identifier, args.json_arguments)
