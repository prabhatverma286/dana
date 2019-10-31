import tkinter as tk
import tkinter.ttk as ttk
import cv2
import PIL.Image, PIL.ImageTk
import threading
from environment_simulator import make_environment

Games = [
    "Cartpole",
    "Pong",
    "Breakout"
]  # etc


def start_training():
    print("Hey, this works!")
    training_thread = threading.Thread(target=make_environment, args=["CartPole-v0"])
    training_thread.start()


class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        self.select_game_label = ttk.Label(master=window, text="Select the game")
        self.select_game_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=20)

        self.select_game_value = tk.StringVar(window)
        self.select_game_value.set(Games[0])  # default value

        self.select_game_menu = ttk.OptionMenu(window, self.select_game_value, *Games)
        self.select_game_menu.grid(row=0, column=1, sticky=tk.W, padx=10, pady=20)
        # self.select_game_menu.pack(side=ttk.LEFT, padx=20, pady=20)

        self.start_training_button = ttk.Button(window, text="Start training", command=start_training)
        self.start_training_button.grid(row=0, column=2, sticky=tk.E, padx=10, pady=20)
        # self.start_training_button.pack(side=ttk.RIGHT, padx=20, pady=20)

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(master=window, width=self.vid.width, height=self.vid.height)
        self.canvas.grid(row=2, column=2, padx=20, pady=20)
        # self.canvas.pack(side=ttk.BOTTOM)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update(master=window)

        self.window.mainloop()

    def update(self, master):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(master=master, image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update, master)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return ret, None
        else:
            return None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Create a window and pass it to the Application object
App(tk.Tk(), "Dana - a DQN AgeNt for Atari", "C:\\Users\\prabh\\Videos\\test_video.mp4")
