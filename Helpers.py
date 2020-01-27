import glob
import json
import os
import subprocess


def get_latest_video_from_output_dir(directory_to_scan):
    if directory_to_scan is None:
        return None, None
    list_of_mp4_files = glob.glob(directory_to_scan + "\\*.mp4")
    latest_modified_mp4_files = sorted(list_of_mp4_files, key=os.path.getctime, reverse=True)
    if len(latest_modified_mp4_files) < 3:
        return None, None
    video_to_display = latest_modified_mp4_files[2]
    meta_json = video_to_display[:-4] + ".meta.json"
    with open(meta_json, "r") as fd:
        episode_number = json.load(fd)["episode_id"]
    return latest_modified_mp4_files[2], directory_to_scan + "\\" + str(episode_number) + "-meta.json"


def get_training_dir(env_id):
    return os.path.join('agent-training', env_id)


def get_evaluation_dir(env_id):
    return os.path.join('agent-evaluation', env_id)
