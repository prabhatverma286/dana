import json
import os

import gym
from gym import wrappers, logger
import argparse

from agent import DQNAgent


def save_video_and_stats(episode_number):
    return episode_number % 25 == 0 and episode_number is not 0


def main(env_id, arguments):
    logger.set_level(logger.INFO)
    env = gym.make(env_id)

    outdir = 'agent-evaluation'
    env = wrappers.Monitor(env, directory=outdir, force=True, video_callable=lambda x: True)
    env.seed(0)

    agent = DQNAgent(outdir, env)
    agent.from_path(arguments, "saved_models\\" + env_id + "_model.pkl")

    for i in range(100):
        ob = env.reset()
        total_reward = 0
        while True:
            action = agent.take_action(ob)
            ob, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                file_name = os.path.join(outdir, str(i) + '-meta.json')
                with open(file_name, 'w') as fd:
                    fd.write(json.dumps({
                        'episode_number': i,
                        'episode_score': total_reward
                    }))
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', help='gym env id')
    parser.add_argument('--json_arguments', type=json.loads, help='gym env id')

    args = parser.parse_args()
    main(args.env_id, args.json_arguments)
