import json
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk as ttk

import PIL.Image
import PIL.ImageTk
import cv2

from Helpers import get_training_dir, get_evaluation_dir, get_latest_video_from_output_dir

env = "CartPole-v0"
Games = [
    "Cartpole",
    "Pong",
    "Breakout"
]  # etc


class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_directory = None
        self.video_source = None
        self.vid_playing = None
        self.video_metadata = None
        self.training_process = None
        self.evaluation_process = None

        self.select_game_label = ttk.Label(master=window, text="Select the game")
        self.select_game_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=30)

        self.select_game_value = tk.StringVar(window)
        self.select_game_value.set(Games[0])  # default value

        self.select_game_menu = ttk.OptionMenu(window, self.select_game_value, None, *Games)
        self.select_game_menu.grid(row=0, column=1, sticky=tk.W, padx=10, pady=30)

        self.start_training_button = ttk.Button(window, text="Start training", command=self.start_training)
        self.start_training_button.grid(row=0, column=2, sticky=tk.E, padx=(20, 5), pady=30)

        self.evaluation_button = ttk.Button(window, text="Evaluate agent", command=self.evaluate)
        self.evaluation_button.grid(row=0, column=3, sticky=tk.E, padx=(5, 20), pady=30)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(master=window, width=600, height=600)
        self.canvas.grid(row=1, rowspan=100, columnspan=100, column=3, padx=20, pady=20, sticky=tk.E)

        self.episode_number_label = tk.Label(master=window, text="Episode number:")
        self.episode_number_label.grid(row=1, column=0, padx=20, pady=(30, 5), sticky=tk.W)

        self.episode_reward_label = tk.Label(master=window, text="Episode reward:")
        self.episode_reward_label.grid(row=2, column=0, padx=20, pady=5, sticky=tk.W)

        self.episode_number_text = tk.Label(master=window, text="-")
        self.episode_number_text.grid(row=1, column=1, padx=20, pady=(30, 5), sticky=tk.W)

        self.episode_reward_text = tk.Label(master=window, text="-")
        self.episode_reward_text.grid(row=2, column=1, padx=20, pady=5, sticky=tk.W)

        self.training_or_evaluating_text = tk.Label(master=window, text="-")
        self.training_or_evaluating_text.grid(row=3, column=2, padx=20, pady=5, sticky=tk.W)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update(master=window)

        self.window.mainloop()

    def update(self, master):
        if self.vid_playing is None:
            self.video_source, self.video_metadata = get_latest_video_from_output_dir(self.video_directory)
            if self.video_source is None:
                self.photo = tk.PhotoImage(file="resources\\default_image.png")
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.episode_number_text.text = "-"
                self.episode_reward_text.text = "-"
            else:
                self.vid_playing = VideoReader(self.video_source)
                with open(self.video_metadata, "r") as fd:
                    metadata = json.load(fd)

                self.episode_number_text['text'] = metadata["episode_number"]
                self.episode_reward_text['text'] = metadata["episode_score"]
        else:
            # open the found video
            # if video open
            #   if get next frame
            #       show next frame and exit
            #   else
            #       exit and find new video in next update
            ret, frame = self.vid_playing.get_frame()
            if ret:
                self.photo = PIL.ImageTk.PhotoImage(master=master, image=PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            else:
                self.vid_playing = None
                self.video_source = None

        if self.training_process is not None and self.training_process.poll() is None:
            self.training_or_evaluating_text['text'] = "TRAINING"
        elif self.evaluation_process is not None and self.evaluation_process.poll() is None:
            self.training_or_evaluating_text['text'] = "EVALUATING"
        else:
            self.training_or_evaluating_text['text'] = "-"

        self.window.after(self.delay, self.update, master)

    def evaluate(self):
        self.video_directory = get_evaluation_dir(env)
        self.vid_playing = None
        self.video_source = None

        self.evaluation_process = subprocess.Popen([sys.executable, 'evaluate.py', '--env_id', env])

    def start_training(self):
        self.video_directory = get_training_dir(env)
        self.vid_playing = None
        self.video_source = None

        self.training_process = subprocess.Popen([sys.executable, 'train.py', '--env_id', env])


class VideoReader:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                self.vid.release()
                return ret, None
        else:
            return None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
