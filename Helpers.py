import glob
import json
import os
import subprocess
from datetime import datetime

from gym import wrappers


def get_latest_video_from_output_dir(directory_to_scan):
    if directory_to_scan is None:
        return None, None
    list_of_mp4_files = glob.glob(directory_to_scan + "\\*.mp4")
    latest_modified_mp4_files = sorted(list_of_mp4_files, key=os.path.getctime, reverse=True)
    if len(latest_modified_mp4_files) < 2:
        return None, None
    video_to_display = latest_modified_mp4_files[1]
    meta_json = video_to_display[:-4] + ".meta.json"
    with open(meta_json, "r") as fd:
        episode_number = json.load(fd)["episode_id"]
    if episode_number == 0:
        return None, None
    return latest_modified_mp4_files[1], directory_to_scan + "\\" + str(episode_number) + "-meta.json"


def get_training_dir(env_id):
    return os.path.join('agent-training', env_id)


def get_evaluation_dir(env_id):
    return os.path.join('agent-evaluation', env_id)


class BreakoutMonitor(wrappers.Monitor):
    def __init__(self, env, directory, **kwargs):
        super().__init__(env, directory, **kwargs)

    def step(self, action):
        self._before_step(action)
        observation, reward, done, info = self.env.step(action)
        if done and info['ale.lives'] > 0:
            done = False

        done = self._after_step(observation, reward, done, info)

        return observation, reward, done, info


envs = {
    "Cartpole": "CartPole-v0",
    "Pong": "PongNoFrameskip-v4",
    "Breakout": "BreakoutNoFrameskip-v4"
}