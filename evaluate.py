import json
import os

import gym
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, WarpFrame, FrameStack
from baselines.common.retro_wrappers import make_retro
from gym import logger
import argparse

from gym.wrappers import Monitor

from Helpers import BreakoutMonitor, envs, disable_view_window
from agent import DQNAgent
from keras_dqfd import SonicDiscretizer, AllowBacktracking


def main(env_id, arguments):
    logger.set_level(logger.INFO)

    if env_id == envs["Cartpole"]:
        disable_view_window()
        env = gym.make(env_id)
    elif env_id == envs["Sonic_the_HedgeHog"]:
        env = make_retro(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', scenario='contest')

        action_list = [['NOOP'], ['LEFT'], ['RIGHT'], ['LEFT', 'B'], ['RIGHT', 'B'], ['DOWN'], ['B']]

        env = SonicDiscretizer(env, action_list)
        env = AllowBacktracking(env)
        env = WarpFrame(env)
        env = FrameStack(env, 4)
    else:
        env = make_atari(env_id)
        env = wrap_deepmind(env, frame_stack=True, scale=False, clip_rewards=False)

    out_dir = 'agent-evaluation'
    os.makedirs(out_dir, exist_ok=True)

    if env_id == envs["Breakout"]:
        # Wrapper for episodic life in breakout
        env = BreakoutMonitor(env, directory=out_dir, force=True, video_callable=lambda x: True)
    else:
        env = Monitor(env, directory=out_dir, force=True, video_callable=lambda x: True)

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
                file_name = os.path.join(out_dir, str(i+1) + '-meta.json')
                with open(file_name, 'w') as fd:
                    fd.write(json.dumps({
                        'episode_number': i+1,
                        'episode_score': total_reward
                    }))
                break
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', help='gym env id')
    parser.add_argument('--json_arguments', type=json.loads, help='json dictionary of the training arguments')

    args = parser.parse_args()
    main(args.env_id, args.json_arguments)
