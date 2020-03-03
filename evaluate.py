import json
import os

import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from gym import logger
import argparse

from gym.wrappers import Monitor

from Helpers import BreakoutMonitor, envs
from agent import DQNAgent


def save_video_and_stats(episode_number):
    if episode_number == 1:
        return True
    return episode_number % 5 == 0 and episode_number is not 0
    # return True


def main(env_id, arguments):
    logger.set_level(logger.INFO)

    if env_id == envs["Cartpole"]:
        env = gym.make(env_id)
    else:
        env = make_atari(env_id)
        env = wrap_deepmind(env, frame_stack=True, scale=False, clip_rewards=False)

    out_dir = 'agent-evaluation'
    os.makedirs(out_dir, exist_ok=True)

    if env_id == envs["Breakout"]:
        env = BreakoutMonitor(env, directory=out_dir, force=True, video_callable=save_video_and_stats)
    else:
        env = Monitor(env, directory=out_dir, force=True, video_callable=save_video_and_stats)

    env.seed(0)

    agent = DQNAgent(out_dir, env)
    agent.from_path(arguments, "saved_models\\" + env_id + "_model.pkl")

    for i in range(100):
        ob = env.reset()
        total_reward = 0
        while True:
            action = agent.take_action(ob)
            ob, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                file_name = os.path.join(out_dir, str(i + 1) + '-meta.json')
                with open(file_name, 'w') as fd:
                    fd.write(json.dumps({
                        'episode_number': i + 1,
                        'episode_score': total_reward
                    }))
                break
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', help='gym env id')
    parser.add_argument('--json_arguments', type=json.loads, help='gym env id')

    args = parser.parse_args()
    main(args.env_id, args.json_arguments)
