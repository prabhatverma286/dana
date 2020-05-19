import json
import os
import numpy as np

from baselines import bench
from baselines.common.atari_wrappers import WarpFrame, FrameStack
from baselines.common.retro_wrappers import make_retro
from baselines import logger
import argparse

from gym.wrappers import Monitor

from Helpers import BreakoutMonitor, envs
from agent import DQNAgent
from keras_dqfd import build_model, AllowBacktracking, SonicDiscretizer


def main(env_id, arguments):
    logger.set_level(logger.INFO)

    env = make_retro(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', scenario='contest')
    env = bench.Monitor(env, logger.get_dir())

    action_list = [['NOOP'], ['LEFT'], ['RIGHT'], ['LEFT', 'B'], ['RIGHT', 'B'], ['DOWN'], ['B']]

    env = SonicDiscretizer(env, action_list)
    env = AllowBacktracking(env)

    env = WarpFrame(env)
    env = FrameStack(env, 4)

    out_dir = 'agent-evaluation'
    os.makedirs(out_dir, exist_ok=True)

    if env_id == envs["Breakout"]:
        # Wrapper for episodic life in breakout
        env = BreakoutMonitor(env, directory=out_dir, force=True, video_callable=lambda x: True)
    else:
        env = Monitor(env, directory=out_dir, force=True, video_callable=lambda x: True)

    env.seed(0)

    model = build_model(7)
    model.load_weights("saved_models\\" + envs["Sonic_the_HedgeHog"] + "_dqfd_model.h5")

    empty_by_one = np.zeros((1, 1))
    empty_exp_action_by_one = np.zeros((1, 2))
    empty_action_len_by_one = np.zeros((1, 7))

    for i in range(100):
        ob = env.reset()
        total_reward = 0
        while True:
            temp_curr_obs = np.array(ob)
            temp_curr_obs = temp_curr_obs.reshape(1, temp_curr_obs.shape[0], temp_curr_obs.shape[1],
                                                  temp_curr_obs.shape[2])
            action, _, _ = model.predict([temp_curr_obs, temp_curr_obs,empty_by_one, empty_exp_action_by_one,empty_action_len_by_one])
            action = np.argmax(action)
            ob, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                file_name = os.path.join(out_dir, str(i) + '-meta.json')
                with open(file_name, 'w') as fd:
                    fd.write(json.dumps({
                        'episode_number': i,
                        'episode_score': total_reward
                    }))
                print("reward: {}".format(total_reward))
                break
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', help='gym env id')
    parser.add_argument('--json_arguments', type=json.loads, help='json dictionary of the training arguments')

    args = parser.parse_args()
    main(args.env_id, args.json_arguments)
