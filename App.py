import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from pprint import pprint, pformat
from tkinter import ttk as ttk
import tkinter.scrolledtext as scrolledText

import PIL.Image
import PIL.ImageTk
import cv2

import default_params
from Helpers import get_latest_video_from_output_dir, envs

default_model_text = "No pre-trained models found"


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

        self.window.minsize(width=300, height=15)

        self.select_game_label = ttk.Label(master=window, text="Environment")
        self.select_game_label.grid(row=0, column=0, padx=(30, 10), pady=(30, 5))

        self.select_game_value = tk.StringVar(window)
        self.select_game_value.set("-")  # default value

        self.select_game_menu = ttk.OptionMenu(window, self.select_game_value, None, *envs.keys())
        self.select_game_menu.grid(row=0, column=1, padx=10, pady=(30, 5))

        self.select_game_button = ttk.Button(master=window, text="Select", command=self.select_environment)
        self.select_game_button.grid(row=0, column=2, padx=10, pady=(30, 5))

        self.parameters_text_box = scrolledText.ScrolledText(window, height=10, width=40)

        self.mode = tk.IntVar()
        self.start_training_radio = ttk.Radiobutton(master=window, text="Training", variable=self.mode, value=1)
        self.start_training_radio.config(command=self.toggle_modes)
        self.start_evaluating_radio = ttk.Radiobutton(master=window, text="Evaluation", variable=self.mode, value=2)
        self.start_evaluating_radio.config(command=self.toggle_modes)

        self.start_training_button = ttk.Button(window, text="Start training", command=self.start_training)

        self.trained_model_label = ttk.Label(window, text=default_model_text)

        self.evaluation_button = ttk.Button(window, text="Evaluate agent", command=self.evaluate)

        self.reset_button = ttk.Button(window, text="Stop and Reset", command=self.stop_and_reset)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(master=window, width=160, height=210)

        self.episode_number_label = tk.Label(master=window, text="Episode number:")

        self.episode_reward_label = tk.Label(master=window, text="Episode reward:")

        self.training_or_evaluating_text = tk.Label(master=window, text="-")

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update(master=window)

        self.window.mainloop()

    def update(self, master):
        self.window.after(self.delay, self.update, master)

        if self.evaluation_process is None and self.training_process is None:
            return

        if self.vid_playing is None:
            self.video_source, self.video_metadata = get_latest_video_from_output_dir(self.video_directory)
            if self.video_source is not None:
                self.canvas.grid(row=2, rowspan=100, columnspan=100, column=3, padx=10, pady=5, sticky='nsew')
                self.episode_number_label.grid(row=0, columnspan=100, column=3, padx=10, pady=5, sticky='nsew')
                self.episode_reward_label.grid(row=1, columnspan=100, column=3, padx=10, pady=5, sticky='nsew')

                self.vid_playing = VideoReader(self.video_source)
                with open(self.video_metadata, "r") as fd:
                    metadata = json.load(fd)

                if self.training_process is not None and self.training_process.poll() is None:
                    self.training_or_evaluating_text['text'] = "Training.."
                elif self.evaluation_process is not None and self.evaluation_process.poll() is None:
                    self.training_or_evaluating_text['text'] = "Evaluating.."

                self.episode_number_label['text'] = "Episode number:" + "\t" + str(metadata["episode_number"])
                self.episode_reward_label['text'] = "Episode reward:" + "\t" + str(metadata["episode_score"])
        else:
            ret, frame = self.vid_playing.get_frame()
            if ret:
                self.photo = PIL.ImageTk.PhotoImage(master=master,
                                                    image=PIL.Image.fromarray(frame).resize((160, 210),
                                                                                            PIL.Image.ANTIALIAS))
                self.canvas.config(width=self.photo.width(), height=self.photo.height())
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            else:
                self.vid_playing = None
                self.video_source = None

        if (self.training_process is not None and self.training_process.poll() is not None) or \
                (self.evaluation_process is not None and self.evaluation_process.poll() is not None):
            self.training_process = None
            self.evaluation_process = None
            self.vid_playing = None
            self.video_source = None
            self.canvas.grid_forget()
            self.episode_number_label.grid_forget()
            self.episode_reward_label.grid_forget()
            self.training_or_evaluating_text.grid_forget()
            self.reset_button.grid_forget()
            self.toggle_controls(enabled=True)

    def evaluate(self):
        if os.path.exists('agent-evaluation'):
            shutil.rmtree('agent-evaluation')

        self.video_directory = 'agent-evaluation'
        self.vid_playing = None
        self.video_source = None
        self.training_or_evaluating_text['text'] = "Setting up.."

        self.training_or_evaluating_text.grid(row=4, column=1, padx=(10, 20), pady=5)
        self.reset_button.grid(row=5, column=1, padx=(10, 20), pady=5, sticky=tk.W)

        self.toggle_controls(enabled=False)

        selected_env = self.select_game_value.get()
        if selected_env.__eq__("-"):
            return

        params = getattr(default_params, selected_env.lower()) # json.loads(self.parameters_text_box.get("1.0", tk.END))

        self.evaluation_process = subprocess.Popen([sys.executable, 'evaluate.py', '--env_id', envs[selected_env], '--json_arguments', json.dumps(params)])

    def start_training(self):
        if os.path.exists('agent-training'):
            shutil.rmtree('agent-training')

        self.video_directory = 'agent-training'
        self.vid_playing = None
        self.video_source = None
        self.training_or_evaluating_text['text'] = "Setting up.."

        self.training_or_evaluating_text.grid(row=4, column=1, padx=(10, 20), pady=5)
        self.reset_button.grid(row=5, column=1, padx=(10, 20), pady=5, sticky=tk.W)

        self.toggle_controls(enabled=False)

        params = json.loads(self.parameters_text_box.get("1.0", tk.END))

        selected_env = self.select_game_value.get()
        if selected_env.__eq__("-"):
            return

        self.training_process = subprocess.Popen([sys.executable, 'train.py', '--env_id', envs[selected_env], '--json_arguments', json.dumps(params)])

    def stop_and_reset(self):
        if self.training_process is not None:
            self.training_process.kill()
        if self.evaluation_process is not None:
            self.evaluation_process.kill()

        self.vid_playing = None
        self.video_source = None
        self.canvas.grid_forget()
        self.episode_number_label.grid_forget()
        self.episode_reward_label.grid_forget()
        self.training_or_evaluating_text.grid_forget()
        self.reset_button.grid_forget()
        self.toggle_controls(True)

    def toggle_controls(self, enabled):
        if enabled:
            self.start_training_button.config(state='normal')
            self.evaluation_button.config(state='normal')
            self.select_game_menu.config(state='normal')
            self.select_game_button.config(state='normal')
            self.parameters_text_box.config(state='normal')
            self.start_training_radio.config(state='normal')
            self.start_evaluating_radio.config(state='normal')

        else:
            self.start_training_button.config(state='disabled')
            self.evaluation_button.config(state='disabled')
            self.select_game_menu.config(state='disabled')
            self.select_game_button.config(state='disabled')
            self.parameters_text_box.config(state='disabled')
            self.start_training_radio.config(state='disabled')
            self.start_evaluating_radio.config(state='disabled')

    def select_environment(self):
        selected_env = self.select_game_value.get()
        if selected_env.__eq__("-"):
            return
        self.start_evaluating_radio.grid(row=1, column=0, padx=(50, 0), pady=5)
        self.start_training_radio.grid(row=1, column=2, padx=(0, 50), pady=5)

    def toggle_modes(self):
        selected_env = self.select_game_value.get()
        if selected_env.__eq__("-"):
            return

        if self.mode.get() == 1:
            self.parameters_text_box.grid(row=2, column=0, padx=30, pady=5, columnspan=3)
            self.parameters_text_box.delete("1.0", tk.END)
            self.parameters_text_box.insert(tk.INSERT, json.dumps(getattr(default_params, selected_env.lower()), indent=4))
            self.start_training_button.grid(row=3, column=0, padx=(10, 20), pady=5, columnspan=3)

            self.trained_model_label.grid_forget()
            self.evaluation_button.grid_forget()

        elif self.mode.get() == 2:
            self.trained_model_label.grid(row=2, column=0, pady=5, columnspan=3)

            models = glob.glob("saved_models\\*" + selected_env + "*")
            if len(models) > 0:
                self.trained_model_label['text'] = models[0].strip("saved_models\\")
            else:
                self.trained_model_label['text'] = default_model_text

            self.evaluation_button.grid(row=3, column=0, padx=(10, 20), pady=5, columnspan=3)
            if self.trained_model_label['text'].__eq__(default_model_text):
                self.evaluation_button.config(state='disabled')

            self.parameters_text_box.grid_forget()
            self.start_training_button.grid_forget()


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
