import glob
import json
import os
import tkinter as tk
import tkinter.ttk as ttk
import cv2
import PIL.Image, PIL.ImageTk
import threading
from environment_simulator import make_environment, evaluate_agent, get_evaluation_dir, get_training_dir

env = "CartPole-v0"
Games = [
    "Cartpole",
    "Pong",
    "Breakout"
]  # etc


class App:
    def __init__(self, window, window_title, video_directory):
        self.window = window
        self.window.title(window_title)
        self.video_directory = video_directory
        self.video_source = None
        self.vid_playing = None
        self.video_metadata = None

        self.select_game_label = ttk.Label(master=window, text="Select the game")
        self.select_game_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=30)

        self.select_game_value = tk.StringVar(window)
        self.select_game_value.set(Games[0])  # default value

        self.select_game_menu = ttk.OptionMenu(window, self.select_game_value, *Games)
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

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update(master=window)

        self.window.mainloop()

    def update(self, master):
        if self.vid_playing is None:
            self.video_source, self.video_metadata = self.get_latest_video_from_output_dir(self.video_directory)
            if self.video_source is None:
                self.photo = tk.PhotoImage(file="resources\\default_image.png")
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                self.episode_number_text.text = "-"
                self.episode_reward_text.text = "-"
            else:
                self.vid_playing = MyVideoCapture(self.video_source)
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

        self.window.after(self.delay, self.update, master)

    def evaluate(self):
        self.video_directory = get_evaluation_dir(env)
        self.vid_playing = None
        self.video_source = None

        evaluation_thread = threading.Thread(target=evaluate_agent, args=[env, 100])
        evaluation_thread.start()

    def start_training(self):
        print("Hey, this works!")
        training_thread = threading.Thread(target=make_environment, args=[env])
        training_thread.start()

    def get_latest_video_from_output_dir(self, directory_to_scan):
        list_of_mp4_files = glob.glob(directory_to_scan + "\\*.mp4")
        latest_modified_mp4_files = sorted(list_of_mp4_files, key=os.path.getctime, reverse=True)
        if len(latest_modified_mp4_files) < 3:
            return None, None
        video_to_display = latest_modified_mp4_files[2]
        meta_json = video_to_display[:-4] + ".meta.json"
        with open(meta_json, "r") as fd:
            episode_number = json.load(fd)["episode_id"]
        return latest_modified_mp4_files[2], directory_to_scan + "\\" + str(episode_number) + "-meta.json"


class MyVideoCapture:
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


# Create a window and pass it to the Application object
App(tk.Tk(), "Dana - a DQN AgeNt for Atari", get_training_dir(env))
