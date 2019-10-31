import argparse
import os

from agent import RandomAgent, DQNAgent, save_video_and_stats

import gym
from gym import wrappers, logger


def make_environment(env_id):

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
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

    episode_count = 100
    reward = 0
    done = False

    agent.train()

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()
