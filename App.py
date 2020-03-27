import glob
import json
import os
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import ttk as ttk
from tkinter.scrolledtext import ScrolledText

import PIL.Image
import PIL.ImageTk
import cv2

import default_params
from Helpers import get_latest_video_from_output_dir, envs

default_model_text = "No pre-trained models found"


class App:
    def __init__(self, window, window_title):
        # Initialize all the labels, fields, frames and buttons
        self.window = window
        self.window.title(window_title)
        self.video_directory = None
        self.video_source = None
        self.vid_playing = None
        self.video_metadata = None
        self.training_process = None
        self.evaluation_process = None

        self.window.minsize(width=300, height=15)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        self.window.resizable(False, False)

        # two tabbed notebook
        self.main_nb = ttk.Notebook(master=self.window)
        self.dqn_frame = ttk.Frame(master=self.main_nb)
        self.main_nb.add(self.dqn_frame, text="DQN")

        self.main_nb.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        # #####################################################DQN TAB ###############################################
        # Defining frames #################################################################################
        self.game_selection_frame = ttk.LabelFrame(master=self.dqn_frame, height=1, border=1, text='Configuration')
        self.game_selection_frame.grid(row=0, padx=(10, 10), pady=(30, 5), sticky='nsew')

        self.parameters_label_frame = ttk.LabelFrame(master=self.dqn_frame, text="Training Parameters", border=1)

        # create scroll bar around parameters
        self.parameter_canvas = tk.Canvas(self.parameters_label_frame, highlightthickness=0)
        self.parameter_canvas.grid(row=0, column=0, sticky='news')

        self.parameters_label_frame_scroll = tk.Scrollbar(self.parameters_label_frame, orient=tk.VERTICAL,
                                                          command=self.parameter_canvas.yview)
        self.parameters_label_frame_scroll.grid(row=0, column=1, sticky='ns')
        self.parameter_canvas.configure(yscrollcommand=self.parameters_label_frame_scroll.set)

        self.parameters_frame = tk.Frame(self.parameter_canvas)
        self.parameter_canvas.create_window((0, 0), window=self.parameters_frame)

        def on_frame_configure(event):
            self.parameter_canvas.configure(scrollregion=self.parameter_canvas.bbox("all"))

        def _on_mousewheel(event):
            self.parameter_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bound_to_mousewheel(event):
            self.parameter_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbound_to_mousewheel(event):
            self.parameter_canvas.unbind_all("<MouseWheel>")

        self.parameters_frame.bind("<Configure>", on_frame_configure)
        self.parameters_frame.bind('<Enter>', _bound_to_mousewheel)
        self.parameters_frame.bind('<Leave>', _unbound_to_mousewheel)

        self.trained_model_frame = ttk.LabelFrame(master=self.dqn_frame, height=1, text="Trained Model Name", border=1)
        self.agent_performance_frame = ttk.LabelFrame(master=self.dqn_frame, text="Agent Performance", border=1)

        # Configuration frame elements ####################################################################
        self.select_game_label = ttk.Label(master=self.game_selection_frame, text="Environment")
        self.select_game_label.grid(row=0, column=0, padx=100, pady=5, sticky=tk.W)

        self.select_game_value = tk.StringVar(self.dqn_frame)
        self.select_game_value.set("No Environment Selected")  # default value

        self.select_game_value.trace_variable('w', callback=self.select_environment)
        self.select_game_menu = ttk.OptionMenu(self.game_selection_frame, self.select_game_value, None, *envs.keys())
        self.select_game_menu.grid(row=0, column=1, padx=100, pady=5, sticky=tk.W)

        self.mode = tk.IntVar()
        self.start_training_radio = ttk.Radiobutton(master=self.game_selection_frame, text="Training",
                                                    variable=self.mode, value=1)
        self.start_training_radio.config(command=self.toggle_modes)
        self.start_evaluating_radio = ttk.Radiobutton(master=self.game_selection_frame, text="Evaluation",
                                                      variable=self.mode, value=2)
        self.start_evaluating_radio.config(command=self.toggle_modes)

        # Training Parameters Labels and Entry boxes #########################################################
        self.network_param_label = ttk.Label(master=self.parameters_frame, text="network")
        self.network_param_value = ttk.Entry(master=self.parameters_frame)

        self.seed_param_label = ttk.Label(master=self.parameters_frame, text="seed")
        self.seed_param_value = ttk.Entry(master=self.parameters_frame)

        self.lr_param_label = ttk.Label(master=self.parameters_frame, text="learning rate")
        self.lr_param_value = ttk.Entry(master=self.parameters_frame)

        self.total_timesteps_param_label = ttk.Label(master=self.parameters_frame, text="total timesteps")
        self.total_timesteps_param_value = ttk.Entry(master=self.parameters_frame)

        self.buffer_size_param_label = ttk.Label(master=self.parameters_frame, text="buffer size")
        self.buffer_size_param_value = ttk.Entry(master=self.parameters_frame)

        self.exploration_fraction_param_label = ttk.Label(master=self.parameters_frame, text="exploration fraction")
        self.exploration_fraction_param_value = ttk.Entry(master=self.parameters_frame)

        self.exploration_final_eps_param_label = ttk.Label(master=self.parameters_frame,
                                                           text="exploration final epsilon")
        self.exploration_final_eps_param_value = ttk.Entry(master=self.parameters_frame)

        self.train_freq_param_label = ttk.Label(master=self.parameters_frame, text="training frequency")
        self.train_freq_param_value = ttk.Entry(master=self.parameters_frame)

        self.batch_size_param_label = ttk.Label(master=self.parameters_frame, text="batch size")
        self.batch_size_param_value = ttk.Entry(master=self.parameters_frame)

        self.learning_starts_param_label = ttk.Label(master=self.parameters_frame, text="learning starts")
        self.learning_starts_param_value = ttk.Entry(master=self.parameters_frame)

        self.gamma_param_label = ttk.Label(master=self.parameters_frame, text="gamma")
        self.gamma_param_value = ttk.Entry(master=self.parameters_frame)

        self.target_network_update_freq_param_label = ttk.Label(master=self.parameters_frame,
                                                                text="target network update frequency")
        self.target_network_update_freq_param_value = ttk.Entry(master=self.parameters_frame)

        self.prioritized_replay_alpha_param_label = ttk.Label(master=self.parameters_frame, text="PER alpha")
        self.prioritized_replay_alpha_param_value = ttk.Entry(master=self.parameters_frame)

        self.prioritized_replay_beta0_param_label = ttk.Label(master=self.parameters_frame, text="PER beta")
        self.prioritized_replay_beta0_param_value = ttk.Entry(master=self.parameters_frame)

        self.prioritized_replay_eps_param_label = ttk.Label(master=self.parameters_frame, text="PER epsilon")
        self.prioritized_replay_eps_param_value = ttk.Entry(master=self.parameters_frame)

        self.network_kwargs_param_label = ttk.Label(master=self.parameters_frame, text="additional network kwargs")
        self.network_kwargs_param_value = ScrolledText(master=self.parameters_frame, height=4, width=16)

        # Defining boolean parameter checkboxes #############################################################
        self.dueling = tk.IntVar()
        self.dueling_label = ttk.Label(master=self.parameters_frame, text="dueling dqn")
        self.dueling_checkbox = tk.Checkbutton(master=self.parameters_frame, variable=self.dueling)

        self.param_noise = tk.IntVar()
        self.param_noise_label = ttk.Label(master=self.parameters_frame, text="parameter noise")
        self.param_noise_checkbox = tk.Checkbutton(master=self.parameters_frame, variable=self.param_noise)

        self.per = tk.IntVar()
        self.per_label = ttk.Label(master=self.parameters_frame, text="prioritized experience replay")
        self.per_checkbox = tk.Checkbutton(master=self.parameters_frame, variable=self.per)
        self.per_checkbox.config(command=self.toggle_per)

        # Defining parameter value maps #######################################################################

        self.parameter_values_map = {
            'network': self.network_param_value,
            'seed': self.seed_param_value,
            'lr': self.lr_param_value,
            'buffer_size': self.buffer_size_param_value,
            'total_timesteps': self.total_timesteps_param_value,
            'exploration_final_eps': self.exploration_final_eps_param_value,
            'exploration_fraction': self.exploration_fraction_param_value,
            'train_freq': self.train_freq_param_value,
            'batch_size': self.batch_size_param_value,
            'learning_starts': self.learning_starts_param_value,
            'gamma': self.gamma_param_value,
            'target_network_update_freq': self.target_network_update_freq_param_value,
            'prioritized_replay_alpha': self.prioritized_replay_alpha_param_value,
            'prioritized_replay_beta0': self.prioritized_replay_beta0_param_value,
            'prioritized_replay_eps': self.prioritized_replay_eps_param_value
        }

        self.boolean_parameter_value_map = {
            'dueling': self.dueling,
            'param_noise': self.param_noise,
            'prioritized_replay': self.per
        }

        self.per_parameter_value_map = {
            'prioritized_replay_alpha': self.prioritized_replay_alpha_param_value,
            'prioritized_replay_beta0': self.prioritized_replay_beta0_param_value,
            'prioritized_replay_eps': self.prioritized_replay_eps_param_value
        }

        # Buttons ################################################################################################

        self.start_training_button = ttk.Button(self.dqn_frame, text="Start training", command=self.start_training)
        self.evaluation_button = ttk.Button(self.dqn_frame, text="Evaluate agent", command=self.evaluate)
        self.reset_button = ttk.Button(self.dqn_frame, text="Stop and Reset", command=self.stop_and_reset)

        self.trained_model_label = ttk.Label(self.trained_model_frame, text=default_model_text)

        # Canvas to show the video ################################################################################
        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(master=self.agent_performance_frame, width=320, height=350)
        self.episode_number_label = tk.Label(master=self.agent_performance_frame, text="Episode number:")
        self.episode_reward_label = tk.Label(master=self.agent_performance_frame, text="Episode reward:")
        self.training_or_evaluating_text = tk.Label(master=self.dqn_frame, text="-")

        ############################################################################################################
        ###############################################DQFD FRAME###################################################
        self.dqfd_frame = ttk.Frame(master=self.main_nb)
        self.main_nb.add(self.dqfd_frame, text="DQfD")
        self.main_nb.bind("<<NotebookTabChanged>>", self.tab_changed)

        # Defining frames #################################################################################
        self.game_selection_frame_dqfd = ttk.LabelFrame(master=self.dqfd_frame, height=1, border=1,
                                                        text='Configuration')
        self.game_selection_frame_dqfd.grid(row=0, padx=(10, 10), pady=(30, 5), sticky='nsew')

        self.parameters_label_frame_dqfd = ttk.LabelFrame(master=self.dqfd_frame, text="Training Parameters", border=1)

        self.parameter_canvas_dqfd = tk.Canvas(self.parameters_label_frame_dqfd, highlightthickness=0)
        self.parameter_canvas_dqfd.grid(row=0, column=0, sticky='news')

        self.parameters_label_frame_dqfd_scroll = tk.Scrollbar(self.parameters_label_frame_dqfd, orient=tk.VERTICAL,
                                                               command=self.parameter_canvas_dqfd.yview)
        self.parameters_label_frame_dqfd_scroll.grid(row=0, column=1, sticky='ns')
        self.parameter_canvas_dqfd.configure(yscrollcommand=self.parameters_label_frame_dqfd_scroll.set)

        self.parameters_frame_dqfd = tk.Frame(self.parameter_canvas_dqfd, border=1)
        self.parameter_canvas_dqfd.create_window((0, 0), window=self.parameters_frame_dqfd)

        def on_frame_dqfd_configure(event):
            self.parameter_canvas_dqfd.configure(scrollregion=self.parameter_canvas_dqfd.bbox("all"))

        def _on_mousewheel_dqfd(event):
            self.parameter_canvas_dqfd.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bound_to_mousewheel_dqfd(event):
            self.parameter_canvas_dqfd.bind_all("<MouseWheel>", _on_mousewheel_dqfd)

        def _unbound_to_mousewheel_dqfd(event):
            self.parameter_canvas_dqfd.unbind_all("<MouseWheel>")

        self.parameters_frame_dqfd.bind("<Configure>", on_frame_dqfd_configure)
        self.parameters_frame_dqfd.bind('<Enter>', _bound_to_mousewheel_dqfd)
        self.parameters_frame_dqfd.bind('<Leave>', _unbound_to_mousewheel_dqfd)

        self.trained_model_frame_dqfd = ttk.LabelFrame(master=self.dqfd_frame, height=1, text="Trained Model Name",
                                                       border=1)
        self.agent_performance_frame_dqfd = ttk.LabelFrame(master=self.dqfd_frame, text="Agent Performance", border=1)

        # Configuration frame elements ####################################################################
        self.select_game_label_dqfd = ttk.Label(master=self.game_selection_frame_dqfd, text="Environment")
        self.select_game_label_dqfd.grid(row=0, column=0, padx=70, pady=5, sticky=tk.W)

        self.select_game_value_dqfd = tk.StringVar(self.dqfd_frame)
        self.select_game_value_dqfd.set("No Environment Selected")  # default value

        self.select_game_value_dqfd.trace_variable('w', callback=self.select_environment)
        self.select_game_menu_dqfd = ttk.OptionMenu(self.game_selection_frame_dqfd, self.select_game_value_dqfd, None,
                                                    *['Sonic the HedgeHog'])
        self.select_game_menu_dqfd.grid(row=0, column=2, padx=70, pady=5, sticky=tk.E)

        self.mode_dqfd = tk.IntVar()
        self.train_expert_dqfd_radio = ttk.Radiobutton(master=self.game_selection_frame_dqfd,
                                                       text="Pre-train Expert Model",
                                                       variable=self.mode_dqfd, value=1)
        self.train_expert_dqfd_radio.config(command=self.toggle_modes_dqfd)

        self.train_normal_dqfd_radio = ttk.Radiobutton(master=self.game_selection_frame_dqfd, text="Training",
                                                       variable=self.mode_dqfd, value=2)
        self.train_normal_dqfd_radio.config(command=self.toggle_modes_dqfd)

        self.start_evaluating_radio_dqfd = ttk.Radiobutton(master=self.game_selection_frame_dqfd, text="Evaluation",
                                                      variable=self.mode_dqfd, value=3)
        self.start_evaluating_radio_dqfd.config(command=self.toggle_modes_dqfd)

        # Training Parameters Labels and Entry boxes #########################################################
        self.demo_file_param_label = ttk.Label(master=self.parameters_frame_dqfd, text="Demo file name")
        self.demo_file_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.expert_model_name_param_label = ttk.Label(master=self.parameters_frame_dqfd, text="Expert model to load")
        self.expert_model_name_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.lr_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd, text="learning rate")
        self.lr_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.total_timesteps_expert_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd,
                                                                 text="total timesteps")
        self.total_timesteps_expert_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.total_timesteps_normal_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd,
                                                                 text="total timesteps")
        self.total_timesteps_normal_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.buffer_size_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd, text="PER buffer size")
        self.buffer_size_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.exploration_eps_start_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd,
                                                                text="epsilon starting value")
        self.exploration_eps_start_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.exploration_min_eps_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd,
                                                              text="exploration final epsilon")
        self.exploration_min_eps_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.demo_ratio_param_label = ttk.Label(master=self.parameters_frame_dqfd,
                                                text="ratio of expert samples in one batch")
        self.demo_ratio_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.batch_size_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd, text="batch size")
        self.batch_size_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.learning_starts_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd, text="learning starts")
        self.learning_starts_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.gamma_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd, text="gamma")
        self.gamma_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.target_network_update_freq_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd,
                                                                     text="target network update frequency")
        self.target_network_update_freq_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.prioritized_replay_alpha_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd, text="PER alpha")
        self.prioritized_replay_alpha_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.prioritized_replay_beta_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd, text="PER beta")
        self.prioritized_replay_beta_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.prioritized_replay_eps_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd, text="PER epsilon")
        self.prioritized_replay_eps_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.dqn_loss_weight_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd,
                                                          text="weight for dqn loss in loss function")
        self.dqn_loss_weight_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.nstep_loss_weight_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd,
                                                            text="weight for n-step dqn \n   loss in loss function")
        self.nstep_loss_weight_eps_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.slmc_loss_weight_eps_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd,
                                                               text="weight for supervised learning \n"
                                                                    "          classifier in loss "
                                                                    "function")
        self.slmc_loss_weight_eps_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        self.l2_reg_loss_weight_dqfd_param_label = ttk.Label(master=self.parameters_frame_dqfd,
                                                             text="weight for l2 regularization loss")
        self.l2_reg_loss_weight_dqfd_param_value = ttk.Entry(master=self.parameters_frame_dqfd)

        # Defining boolean parameter checkboxes #############################################################
        self.dueling_dqfd = tk.IntVar()
        self.dueling_dqfd_label = ttk.Label(master=self.parameters_frame_dqfd, text="dueling dqn")
        self.dueling_dqfd_checkbox = tk.Checkbutton(master=self.parameters_frame_dqfd, variable=self.dueling_dqfd)

        self.reward_clipping_dqfd = tk.IntVar()
        self.reward_clipping_label = ttk.Label(master=self.parameters_frame_dqfd,
                                               text="clip rewards from the environment")
        self.reward_clipping_checkbox = tk.Checkbutton(master=self.parameters_frame_dqfd,
                                                       variable=self.reward_clipping_dqfd)

        self.allow_backtracking = tk.IntVar()
        self.allow_backtracking_label = ttk.Label(master=self.parameters_frame_dqfd,
                                                  text="allow backtracking for sonic")
        self.allow_backtracking_checkbox = tk.Checkbutton(master=self.parameters_frame_dqfd,
                                                          variable=self.allow_backtracking)
        # Defining parameter value maps #######################################################################

        self.parameter_values_dqfd_map = {
            'demo_file': self.demo_file_param_value,
            'expert_model_name': self.expert_model_name_param_value,
            'lr': self.lr_dqfd_param_value,
            'buffer_size': self.buffer_size_dqfd_param_value,
            'total_timesteps_expert': self.total_timesteps_expert_dqfd_param_value,
            'total_timesteps_normal': self.total_timesteps_normal_dqfd_param_value,
            'exploration_min_eps': self.exploration_min_eps_dqfd_param_value,
            'exploration_eps_start': self.exploration_eps_start_dqfd_param_value,
            'demo_ratio': self.demo_ratio_param_value,
            'batch_size': self.batch_size_dqfd_param_value,
            'learning_starts': self.learning_starts_dqfd_param_value,
            'gamma': self.gamma_dqfd_param_value,
            'target_network_update_freq': self.target_network_update_freq_dqfd_param_value,
            'prioritized_replay_alpha': self.prioritized_replay_alpha_dqfd_param_value,
            'prioritized_replay_beta': self.prioritized_replay_beta_dqfd_param_value,
            'prioritized_replay_eps': self.prioritized_replay_eps_dqfd_param_value,
            'dqn_loss': self.dqn_loss_weight_dqfd_param_value,
            'nstep_loss': self.nstep_loss_weight_eps_dqfd_param_value,
            'slmc_loss': self.slmc_loss_weight_eps_dqfd_param_value,
            'l2_loss': self.l2_reg_loss_weight_dqfd_param_value
        }

        self.boolean_parameter_value_dqfd_map = {
            'dueling': self.dueling_dqfd,
            'reward_clipping': self.reward_clipping_dqfd,
            'allow_backtracking': self.allow_backtracking
        }

        # Buttons ###################################################################################################

        self.start_training_button_dqfd = ttk.Button(self.dqfd_frame, text="Start training",
                                                     command=self.start_training_dqfd)
        self.evaluation_button_dqfd = ttk.Button(self.dqfd_frame, text="Evaluate agent", command=self.evaluate_dqfd)
        self.reset_button_dqfd = ttk.Button(self.dqfd_frame, text="Stop and Reset", command=self.stop_and_reset_dqfd)

        self.trained_model_label_dqfd = ttk.Label(self.trained_model_frame_dqfd, text=default_model_text)

        # Create a canvas that can fit the above video source size
        self.canvas_dqfd = tk.Canvas(master=self.agent_performance_frame_dqfd, width=320, height=350)
        self.episode_number_label_dqfd = tk.Label(master=self.agent_performance_frame_dqfd, text="Episode number:")
        self.episode_reward_label_dqfd = tk.Label(master=self.agent_performance_frame_dqfd, text="Episode reward:")
        self.training_or_evaluating_text_dqfd = tk.Label(master=self.dqfd_frame, text="-")

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update(master=window)

        self.window.mainloop()

    def tab_changed(self, x):
        if self.main_nb.tab(self.main_nb.select(), "text") == 'DQN':
            self.game_selection_frame_dqfd.grid_forget()
            self.parameters_label_frame_dqfd.grid_forget()
            self.agent_performance_frame_dqfd.grid_forget()
            self.trained_model_frame_dqfd.grid_forget()
            self.game_selection_frame.grid(row=0, padx=(10, 10), pady=(30, 5), sticky='nsew')

        elif self.main_nb.tab(self.main_nb.select(), "text") == 'DQfD':
            self.game_selection_frame.grid_forget()
            self.parameters_label_frame.grid_forget()
            self.agent_performance_frame.grid_forget()
            self.trained_model_frame.grid_forget()
            self.game_selection_frame_dqfd.grid(row=0, padx=(10, 10), pady=(30, 5), sticky='nsew')

        self.select_environment(1, 2, 3)

    def update(self, master):
        # gets called every 15 milliseconds, to play the video
        self.window.after(self.delay, self.update, master)

        if self.evaluation_process is None and self.training_process is None:
            return

        # check which tab are we in
        if self.main_nb.tab(self.main_nb.select(), "text") == 'DQN':
            if self.vid_playing is None:
                self.video_source, self.video_metadata = get_latest_video_from_output_dir(self.video_directory)
                if self.video_source is not None:
                    self.agent_performance_frame.grid(row=0, column=2, rowspan=100, padx=(10, 10), pady=(30, 5), sticky='ns')
                    self.canvas.grid(row=2, columnspan=100, column=3, padx=10, pady=5, sticky='sew')
                    self.episode_number_label.grid(row=0, columnspan=100, column=3, padx=10, pady=5, sticky='sew')
                    self.episode_reward_label.grid(row=1, columnspan=100, column=3, padx=10, pady=5, sticky='sew')

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
                                                        image=PIL.Image.fromarray(frame).resize((320, 350),
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
                self.agent_performance_frame.grid_forget()
                self.training_or_evaluating_text.grid_forget()
                self.reset_button.grid_forget()
                self.toggle_controls(enabled=True)

        elif self.main_nb.tab(self.main_nb.select(), "text") == 'DQfD':
            if self.vid_playing is None:
                self.video_source, self.video_metadata = get_latest_video_from_output_dir(self.video_directory)
                if self.video_source is not None:
                    self.agent_performance_frame_dqfd.grid(row=0, column=2, rowspan=100, padx=(10, 10), pady=(30, 5),
                                                      sticky='ns')
                    self.canvas_dqfd.grid(row=2, columnspan=100, column=3, padx=10, pady=5, sticky='sew')
                    self.episode_number_label_dqfd.grid(row=0, columnspan=100, column=3, padx=10, pady=5, sticky='sew')
                    self.episode_reward_label_dqfd.grid(row=1, columnspan=100, column=3, padx=10, pady=5, sticky='sew')

                    self.vid_playing = VideoReader(self.video_source)
                    with open(self.video_metadata, "r") as fd:
                        metadata = json.load(fd)

                    if self.training_process is not None and self.training_process.poll() is None:
                        self.training_or_evaluating_text_dqfd['text'] = "Training.."
                    elif self.evaluation_process is not None and self.evaluation_process.poll() is None:
                        self.training_or_evaluating_text_dqfd['text'] = "Evaluating.."

                    self.episode_number_label_dqfd['text'] = "Episode number:" + "\t" + str(metadata["episode_number"])
                    self.episode_reward_label_dqfd['text'] = "Episode reward:" + "\t" + str(metadata["episode_score"])
            else:
                ret, frame = self.vid_playing.get_frame()
                if ret:
                    self.photo = PIL.ImageTk.PhotoImage(master=master,
                                                        image=PIL.Image.fromarray(frame).resize((320, 350),
                                                                                                PIL.Image.ANTIALIAS))
                    self.canvas_dqfd.config(width=self.photo.width(), height=self.photo.height())
                    self.canvas_dqfd.create_image(0, 0, image=self.photo, anchor=tk.NW)
                else:
                    self.vid_playing = None
                    self.video_source = None

            if (self.training_process is not None and self.training_process.poll() is not None) or \
                    (self.evaluation_process is not None and self.evaluation_process.poll() is not None):
                self.training_process = None
                self.evaluation_process = None
                self.vid_playing = None
                self.video_source = None
                self.agent_performance_frame_dqfd.grid_forget()
                self.training_or_evaluating_text_dqfd.grid_forget()
                self.reset_button_dqfd.grid_forget()
                self.toggle_controls_dqfd(enabled=True)

    def evaluate(self):
        if os.path.exists('agent-evaluation'):
            shutil.rmtree('agent-evaluation')

        self.video_directory = 'agent-evaluation'
        self.vid_playing = None
        self.video_source = None
        self.training_or_evaluating_text['text'] = "Setting up.."

        self.training_or_evaluating_text.grid(row=3, column=0, padx=(10, 20), pady=5)
        self.reset_button.grid(row=4, column=0, padx=(10, 20), pady=5)

        self.toggle_controls(enabled=False)

        selected_env = self.select_game_value.get()
        if selected_env.__eq__("No Environment Selected"):
            return

        params = getattr(default_params,
                         selected_env.lower())  # json.loads(self.parameters_text_box.get("1.0", tk.END))
        for t in self.main_nb.tabs():
            if self.main_nb.tab(t, "text") == 'DQfD':
                self.main_nb.tab(t, state='disabled')
        self.evaluation_process = subprocess.Popen(
            [sys.executable, 'evaluate.py', '--env_id', envs[selected_env], '--json_arguments', json.dumps(params)])

    def start_training(self):
        if os.path.exists('agent-training'):
            shutil.rmtree('agent-training')

        self.video_directory = 'agent-training'
        self.vid_playing = None
        self.video_source = None
        self.training_or_evaluating_text['text'] = "Setting up.."

        self.training_or_evaluating_text.grid(row=3, column=0, padx=(10, 20), pady=5)
        self.reset_button.grid(row=4, column=0, padx=(10, 20), pady=5)

        self.toggle_controls(enabled=False)

        params = {}
        for key, value in self.parameter_values_map.items():
            params[key] = value.get() if value.get() is not '' else None
            if params[key] is not None and key != 'network':
                if key in ['total_timesteps', 'buffer_size', 'batch_size', 'train_freq', 'learning_starts'
                                                                                         'target_network_update_freq']:
                    params[key] = int(params.get(key))
                else:
                    params[key] = float(params.get(key))

        for key, value in self.boolean_parameter_value_map.items():
            params[key] = True if value.get() == 1 else False

        network_kwargs = self.network_kwargs_param_value.get("1.0", tk.END)
        if network_kwargs is not '' and network_kwargs.strip() is not '':
            network_kwargs = json.loads(network_kwargs)
            for k, v in network_kwargs.items():
                params[k] = v

        selected_env = self.select_game_value.get()
        if selected_env.__eq__("No Environment Selected"):
            return

        for t in self.main_nb.tabs():
            if self.main_nb.tab(t, "text") == 'DQfD':
                self.main_nb.tab(t, state='disabled')

        self.training_process = subprocess.Popen(
            [sys.executable, 'train.py', '--env_id', envs[selected_env], '--json_arguments', json.dumps(params)])

    def stop_and_reset(self):
        if self.training_process is not None:
            self.training_process.kill()
        if self.evaluation_process is not None:
            self.evaluation_process.kill()

        self.vid_playing = None
        self.video_source = None
        self.agent_performance_frame.grid_forget()
        self.training_or_evaluating_text.grid_forget()
        self.reset_button.grid_forget()
        self.toggle_controls(True)
        for t in self.main_nb.tabs():
            self.main_nb.tab(t, state='normal')

    def toggle_controls(self, enabled):
        if enabled:
            self.start_training_button.config(state='normal')
            self.evaluation_button.config(state='normal')
            self.select_game_menu.config(state='normal')
            self.start_training_radio.config(state='normal')
            self.start_evaluating_radio.config(state='normal')
            self.enable_params()

        else:
            self.start_training_button.config(state='disabled')
            self.evaluation_button.config(state='disabled')
            self.select_game_menu.config(state='disabled')
            self.start_training_radio.config(state='disabled')
            self.start_evaluating_radio.config(state='disabled')
            self.disable_params()

    def select_environment(self, x, y, z):
        if self.main_nb.tab(self.main_nb.select(), "text") == 'DQN':
            selected_env = self.select_game_value.get()
            if selected_env.__eq__("No Environment Selected"):
                return
            self.start_evaluating_radio.grid(row=1, column=0, padx=10, pady=5)
            self.start_training_radio.grid(row=1, column=1, padx=10, pady=5)
            self.toggle_modes()
        elif self.main_nb.tab(self.main_nb.select(), "text") == 'DQfD':
            selected_env = self.select_game_value_dqfd.get()
            if selected_env.__eq__("No Environment Selected"):
                return
            self.start_evaluating_radio_dqfd.grid(row=1, column=0, pady=5)
            self.train_expert_dqfd_radio.grid(row=1, column=1, pady=5)
            self.train_normal_dqfd_radio.grid(row=1, column=2, pady=5)
            self.toggle_modes_dqfd()

    def toggle_modes(self):
        selected_env = self.select_game_value.get()
        if selected_env.__eq__("-"):
            return

        if self.mode.get() == 1:

            self.parameters_label_frame.grid(row=1, column=0, padx=10, pady=5)
            self.trained_model_frame.grid_forget()
            self.show_params()

            params = getattr(default_params, selected_env.lower())
            self.fill_param_values(params.copy())

            self.start_training_button.grid(row=2, column=0, padx=(10, 20), pady=5)

            self.trained_model_label.grid_forget()
            self.evaluation_button.grid_forget()
            self.parameter_canvas.yview_moveto(0)

        elif self.mode.get() == 2:
            self.trained_model_frame.grid(row=1, column=0, padx=10, pady=5)
            self.parameters_label_frame.grid_forget()
            self.trained_model_label.grid(row=0, column=0, padx=80, pady=5)

            models = glob.glob("saved_models\\*" + envs[selected_env] + "_model.*")
            if len(models) > 0:
                self.trained_model_label['text'] = models[0].split("saved_models\\")[1]
            else:
                self.trained_model_label['text'] = default_model_text

            self.evaluation_button.grid(row=2, column=0, padx=(10, 20), pady=5)
            if self.trained_model_label['text'].__eq__(default_model_text):
                self.evaluation_button.config(state='disabled')

            self.hide_params()
            self.start_training_button.grid_forget()

    def toggle_per(self, force_hide=False):
        if self.per.get() == 0 or force_hide:
            self.prioritized_replay_alpha_param_label.grid_forget()
            self.prioritized_replay_alpha_param_value.grid_forget()

            self.prioritized_replay_beta0_param_label.grid_forget()
            self.prioritized_replay_beta0_param_value.grid_forget()

            self.prioritized_replay_eps_param_label.grid_forget()
            self.prioritized_replay_eps_param_value.grid_forget()
        else:
            self.prioritized_replay_alpha_param_label.grid(row=17, column=0, padx=10, pady=5, sticky=tk.E)
            self.prioritized_replay_alpha_param_value.grid(row=17, column=1, padx=10, pady=5, sticky=tk.W)

            self.prioritized_replay_beta0_param_label.grid(row=18, column=0, padx=10, pady=5, sticky=tk.E)
            self.prioritized_replay_beta0_param_value.grid(row=18, column=1, padx=10, pady=5, sticky=tk.W)

            self.prioritized_replay_eps_param_label.grid(row=19, column=0, padx=10, pady=5, sticky=tk.E)
            self.prioritized_replay_eps_param_value.grid(row=19, column=1, padx=10, pady=5, sticky=tk.W)

    def show_params(self):
        self.network_param_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.E)
        self.network_param_value.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

        self.seed_param_label.grid(row=1, column=0, padx=10, pady=5, sticky=tk.E)
        self.seed_param_value.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)

        self.lr_param_label.grid(row=2, column=0, padx=10, pady=5, sticky=tk.E)
        self.lr_param_value.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)

        self.total_timesteps_param_label.grid(row=3, column=0, padx=10, pady=5, sticky=tk.E)
        self.total_timesteps_param_value.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)

        self.buffer_size_param_label.grid(row=6, column=0, padx=10, pady=5, sticky=tk.E)
        self.buffer_size_param_value.grid(row=6, column=1, padx=10, pady=5, sticky=tk.W)

        self.exploration_fraction_param_label.grid(row=7, column=0, padx=10, pady=5, sticky=tk.E)
        self.exploration_fraction_param_value.grid(row=7, column=1, padx=10, pady=5, sticky=tk.W)

        self.exploration_final_eps_param_label.grid(row=8, column=0, padx=10, pady=5, sticky=tk.E)
        self.exploration_final_eps_param_value.grid(row=8, column=1, padx=10, pady=5, sticky=tk.W)

        self.train_freq_param_label.grid(row=9, column=0, padx=10, pady=5, sticky=tk.E)
        self.train_freq_param_value.grid(row=9, column=1, padx=10, pady=5, sticky=tk.W)

        self.batch_size_param_label.grid(row=10, column=0, padx=10, pady=5, sticky=tk.E)
        self.batch_size_param_value.grid(row=10, column=1, padx=10, pady=5, sticky=tk.W)

        self.learning_starts_param_label.grid(row=11, column=0, padx=10, pady=5, sticky=tk.E)
        self.learning_starts_param_value.grid(row=11, column=1, padx=10, pady=5, sticky=tk.W)

        self.gamma_param_label.grid(row=12, column=0, padx=10, pady=5, sticky=tk.E)
        self.gamma_param_value.grid(row=12, column=1, padx=10, pady=5, sticky=tk.W)

        self.target_network_update_freq_param_label.grid(row=13, column=0, padx=10, pady=5, sticky=tk.E)
        self.target_network_update_freq_param_value.grid(row=13, column=1, padx=10, pady=5, sticky=tk.W)

        self.dueling_label.grid(row=14, column=0, padx=10, pady=5, sticky=tk.E)
        self.dueling_checkbox.grid(row=14, column=1, padx=10, pady=5, sticky=tk.W)

        self.param_noise_label.grid(row=15, column=0, padx=10, pady=5, sticky=tk.E)
        self.param_noise_checkbox.grid(row=15, column=1, padx=10, pady=5, sticky=tk.W)

        self.per_label.grid(row=16, column=0, padx=10, pady=5, sticky=tk.E)
        self.per_checkbox.grid(row=16, column=1, padx=10, pady=5, sticky=tk.W)

        self.network_kwargs_param_label.grid(row=20, column=0, padx=10, pady=5, sticky=tk.E)
        self.network_kwargs_param_value.grid(row=20, column=1, padx=10, pady=5, sticky=tk.W)

    def hide_params(self):
        if self.main_nb.tab(self.main_nb.select(), "text") == 'DQN':
            self.parameters_frame.grid_forget()
        elif self.main_nb.tab(self.main_nb.select(), "text") == 'DQfD':
            self.parameters_frame_dqfd.grid_forget()

    def fill_param_values(self, params):
        params_to_pop = params
        for key, value in self.parameter_values_map.items():
            value_to_insert = params_to_pop.pop(key)
            value.delete(0, tk.END)
            if value_to_insert is not None:
                value.insert(0, value_to_insert)

        for key, value in self.boolean_parameter_value_map.items():
            value_to_insert = params_to_pop.pop(key)
            if value_to_insert:
                value.set(1)
            else:
                value.set(0)
            if key == 'prioritized_replay':
                self.toggle_per()

        if len(params_to_pop) > 0:
            self.network_kwargs_param_value.delete("1.0", tk.END)
            self.network_kwargs_param_value.insert(tk.INSERT, json.dumps(params_to_pop, indent=4))

    def enable_params(self):
        if self.main_nb.tab(self.main_nb.select(), "text") == 'DQN':
            for child in self.parameters_frame.winfo_children():
                if child.widgetName != 'frame':
                    child.config(state='normal')
        elif self.main_nb.tab(self.main_nb.select(), "text") == 'DQfD':
            for child in self.parameters_frame_dqfd.winfo_children():
                child.config(state='normal')

    def disable_params(self):
        if self.main_nb.tab(self.main_nb.select(), "text") == 'DQN':
            for child in self.parameters_frame.winfo_children():
                if child.widgetName != 'frame':
                    child.config(state='disabled')
            self.network_kwargs_param_value.config(state='disabled')
        elif self.main_nb.tab(self.main_nb.select(), "text") == 'DQfD':
            for child in self.parameters_frame_dqfd.winfo_children():
                child.config(state='disabled')

    def evaluate_dqfd(self):
        if os.path.exists('agent-evaluation'):
            shutil.rmtree('agent-evaluation')

        self.video_directory = 'agent-evaluation'
        self.vid_playing = None
        self.video_source = None
        self.training_or_evaluating_text_dqfd['text'] = "Setting up.."

        self.training_or_evaluating_text_dqfd.grid(row=3, column=0, padx=(10, 20), pady=5)
        self.reset_button_dqfd.grid(row=4, column=0, padx=(10, 20), pady=5)

        self.toggle_controls_dqfd(enabled=False)

        selected_env = self.select_game_value_dqfd.get()
        selected_env = selected_env.replace(' ', '_')

        if selected_env.__eq__("No Environment Selected"):
            return

        for t in self.main_nb.tabs():
            if self.main_nb.tab(t, "text") == 'DQN':
                self.main_nb.tab(t, state='disabled')

        params = getattr(default_params,
                         selected_env.lower() + "_dqfd")  # json.loads(self.parameters_text_box.get("1.0", tk.END))

        self.evaluation_process = subprocess.Popen(
            [sys.executable, 'evaluate_keras.py', '--env_id', envs[selected_env], '--json_arguments', json.dumps(params)])

    def start_training_dqfd(self):
        if os.path.exists('agent-training'):
            shutil.rmtree('agent-training')

        self.video_directory = 'agent-training'
        self.vid_playing = None
        self.video_source = None
        self.training_or_evaluating_text_dqfd['text'] = "Setting up.."

        self.training_or_evaluating_text_dqfd.grid(row=3, column=0, padx=(10, 20), pady=5)
        self.reset_button_dqfd.grid(row=4, column=0, padx=(10, 20), pady=5)

        self.toggle_controls_dqfd(enabled=False)

        params = {}
        for key, value in self.parameter_values_dqfd_map.items():
            params[key] = value.get() if value.get() is not '' else None
            if params[key] is not None and key not in ['demo_file', 'expert_model_name']:
                if key in ['total_timesteps_expert', 'total_timesteps_normal', 'buffer_size', 'batch_size', 'train_freq',
                           'learning_starts', 'target_network_update_freq']:
                    params[key] = int(params.get(key))
                else:
                    params[key] = float(params.get(key))

        for key, value in self.boolean_parameter_value_dqfd_map.items():
            params[key] = True if value.get() == 1 else False

        params['train_expert'] = True if self.mode_dqfd.get() == 1 else False

        if self.mode_dqfd.get() == 1:
            self.training_or_evaluating_text_dqfd['text'] = "Training Expert Model... No preview available"

        selected_env = self.select_game_value_dqfd.get()
        if selected_env.__eq__("No Environment Selected"):
            return

        for t in self.main_nb.tabs():
            if self.main_nb.tab(t, "text") == 'DQN':
                self.main_nb.tab(t, state='disabled')

        self.training_process = subprocess.Popen(
            [sys.executable, 'keras_dqfd.py', '--json_arguments', json.dumps(params)])

    def stop_and_reset_dqfd(self):
        if self.training_process is not None:
            self.training_process.kill()
        if self.evaluation_process is not None:
            self.evaluation_process.kill()

        self.vid_playing = None
        self.video_source = None
        self.agent_performance_frame_dqfd.grid_forget()
        self.training_or_evaluating_text_dqfd.grid_forget()
        self.reset_button_dqfd.grid_forget()
        self.toggle_controls_dqfd(True)
        for t in self.main_nb.tabs():
            self.main_nb.tab(t, state='normal')

    def toggle_controls_dqfd(self, enabled):
        if enabled:
            self.start_training_button_dqfd.config(state='normal')
            self.evaluation_button_dqfd.config(state='normal')
            self.select_game_menu_dqfd.config(state='normal')
            self.train_expert_dqfd_radio.config(state='normal')
            self.start_evaluating_radio.config(state='normal')
            self.start_evaluating_radio_dqfd.config(state='normal')
            self.train_normal_dqfd_radio.config(state='normal')
            self.enable_params()

        else:
            self.start_training_button_dqfd.config(state='disabled')
            self.evaluation_button_dqfd.config(state='disabled')
            self.select_game_menu_dqfd.config(state='disabled')
            self.train_expert_dqfd_radio.config(state='disabled')
            self.start_evaluating_radio.config(state='disabled')
            self.start_evaluating_radio_dqfd.config(state='normal')
            self.train_normal_dqfd_radio.config(state='disabled')
            self.disable_params()

    def toggle_modes_dqfd(self):
        selected_env = self.select_game_value_dqfd.get()
        if selected_env.__eq__("-"):
            return
        selected_env = selected_env.replace(' ', '_')

        if self.mode_dqfd.get() == 1 or self.mode_dqfd.get() == 2:

            self.parameters_label_frame_dqfd.grid(row=1, column=0, padx=10, pady=5)
            self.trained_model_frame_dqfd.grid_forget()
            self.show_params_dqfd()

            params = getattr(default_params, selected_env.lower() + "_dqfd")
            self.fill_param_values_dqfd(params)

            self.start_training_button_dqfd.grid(row=2, column=0, padx=(10, 20), pady=5)

            self.trained_model_label_dqfd.grid_forget()
            self.evaluation_button_dqfd.grid_forget()
            self.parameter_canvas_dqfd.yview_moveto(0)

        elif self.mode_dqfd.get() == 3:
            self.trained_model_frame_dqfd.grid(row=1, column=0, padx=10, pady=5)
            self.parameters_label_frame_dqfd.grid_forget()
            self.trained_model_label_dqfd.grid(row=0, column=0, padx=80, pady=5)

            models = glob.glob("saved_models\\*" + envs[selected_env] + "*")
            if len(models) > 0:
                self.trained_model_label_dqfd['text'] = models[0].split("saved_models\\")[1]
            else:
                self.trained_model_label_dqfd['text'] = default_model_text

            self.evaluation_button_dqfd.grid(row=2, column=0, padx=(10, 20), pady=5)
            if self.trained_model_label_dqfd['text'].__eq__(default_model_text):
                self.evaluation_button_dqfd.config(state='disabled')

            self.hide_params()
            self.start_training_button_dqfd.grid_forget()

    def show_params_dqfd(self):
        if self.mode_dqfd.get() == 1:
            self.expert_model_name_param_label.grid_forget()
            self.expert_model_name_param_value.grid_forget()
            self.demo_file_param_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.E)
            self.demo_file_param_value.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)
        elif self.mode_dqfd.get() == 2:
            self.demo_file_param_label.grid_forget()
            self.demo_file_param_value.grid_forget()
            self.expert_model_name_param_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.E)
            self.expert_model_name_param_value.grid(row=0, column=1, padx=10, pady=5, sticky=tk.W)

        self.lr_dqfd_param_label.grid(row=2, column=0, padx=10, pady=5, sticky=tk.E)
        self.lr_dqfd_param_value.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)

        if self.mode_dqfd.get() == 1:
            self.total_timesteps_normal_dqfd_param_label.grid_forget()
            self.total_timesteps_normal_dqfd_param_value.grid_forget()
            self.total_timesteps_expert_dqfd_param_label.grid(row=3, column=0, padx=10, pady=5, sticky=tk.E)
            self.total_timesteps_expert_dqfd_param_value.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)
        elif self.mode_dqfd.get() == 2:
            self.total_timesteps_expert_dqfd_param_label.grid_forget()
            self.total_timesteps_expert_dqfd_param_value.grid_forget()
            self.total_timesteps_normal_dqfd_param_label.grid(row=3, column=0, padx=10, pady=5, sticky=tk.E)
            self.total_timesteps_normal_dqfd_param_value.grid(row=3, column=1, padx=10, pady=5, sticky=tk.W)

        self.buffer_size_dqfd_param_label.grid(row=6, column=0, padx=10, pady=5, sticky=tk.E)
        self.buffer_size_dqfd_param_value.grid(row=6, column=1, padx=10, pady=5, sticky=tk.W)

        self.exploration_eps_start_dqfd_param_label.grid(row=7, column=0, padx=10, pady=5, sticky=tk.E)
        self.exploration_eps_start_dqfd_param_value.grid(row=7, column=1, padx=10, pady=5, sticky=tk.W)

        self.exploration_min_eps_dqfd_param_label.grid(row=8, column=0, padx=10, pady=5, sticky=tk.E)
        self.exploration_min_eps_dqfd_param_value.grid(row=8, column=1, padx=10, pady=5, sticky=tk.W)

        self.demo_ratio_param_label.grid(row=9, column=0, padx=10, pady=5, sticky=tk.E)
        self.demo_ratio_param_value.grid(row=9, column=1, padx=10, pady=5, sticky=tk.W)

        self.batch_size_dqfd_param_label.grid(row=10, column=0, padx=10, pady=5, sticky=tk.E)
        self.batch_size_dqfd_param_value.grid(row=10, column=1, padx=10, pady=5, sticky=tk.W)

        self.learning_starts_dqfd_param_label.grid(row=11, column=0, padx=10, pady=5, sticky=tk.E)
        self.learning_starts_dqfd_param_value.grid(row=11, column=1, padx=10, pady=5, sticky=tk.W)

        self.gamma_dqfd_param_label.grid(row=12, column=0, padx=10, pady=5, sticky=tk.E)
        self.gamma_dqfd_param_value.grid(row=12, column=1, padx=10, pady=5, sticky=tk.W)

        self.target_network_update_freq_dqfd_param_label.grid(row=13, column=0, padx=10, pady=5, sticky=tk.E)
        self.target_network_update_freq_dqfd_param_value.grid(row=13, column=1, padx=10, pady=5, sticky=tk.W)

        self.prioritized_replay_alpha_dqfd_param_label.grid(row=14, column=0, padx=10, pady=5, sticky=tk.E)
        self.prioritized_replay_alpha_dqfd_param_value.grid(row=14, column=1, padx=10, pady=5, sticky=tk.W)

        self.prioritized_replay_beta_dqfd_param_label.grid(row=15, column=0, padx=10, pady=5, sticky=tk.E)
        self.prioritized_replay_beta_dqfd_param_value.grid(row=15, column=1, padx=10, pady=5, sticky=tk.W)

        self.prioritized_replay_eps_dqfd_param_label.grid(row=16, column=0, padx=10, pady=5, sticky=tk.E)
        self.prioritized_replay_eps_dqfd_param_value.grid(row=16, column=1, padx=10, pady=5, sticky=tk.W)

        self.dqn_loss_weight_dqfd_param_label.grid(row=17, column=0, padx=10, pady=5, sticky=tk.E)
        self.dqn_loss_weight_dqfd_param_value.grid(row=17, column=1, padx=10, pady=5, sticky=tk.W)

        self.nstep_loss_weight_dqfd_param_label.grid(row=18, column=0, padx=10, pady=5, sticky=tk.E)
        self.nstep_loss_weight_eps_dqfd_param_value.grid(row=18, column=1, padx=10, pady=5, sticky=tk.W)

        self.slmc_loss_weight_eps_dqfd_param_label.grid(row=19, column=0, padx=10, pady=5, sticky=tk.E)
        self.slmc_loss_weight_eps_dqfd_param_value.grid(row=19, column=1, padx=10, pady=5, sticky=tk.W)

        self.l2_reg_loss_weight_dqfd_param_label.grid(row=20, column=0, padx=10, pady=5, sticky=tk.E)
        self.l2_reg_loss_weight_dqfd_param_value.grid(row=20, column=1, padx=10, pady=5, sticky=tk.W)

        self.dueling_dqfd_label.grid(row=21, column=0, padx=10, pady=5, sticky=tk.E)
        self.dueling_dqfd_checkbox.grid(row=21, column=1, padx=10, pady=5, sticky=tk.W)

        self.reward_clipping_label.grid(row=22, column=0, padx=10, pady=5, sticky=tk.E)
        self.reward_clipping_checkbox.grid(row=22, column=1, padx=10, pady=5, sticky=tk.W)

        self.allow_backtracking_label.grid(row=23, column=0, padx=10, pady=5, sticky=tk.E)
        self.allow_backtracking_checkbox.grid(row=23, column=1, padx=10, pady=5, sticky=tk.W)

    def fill_param_values_dqfd(self, params):
        for key, value in self.parameter_values_dqfd_map.items():
            value_to_insert = params.get(key)
            value.delete(0, tk.END)
            if value_to_insert is not None:
                value.insert(0, value_to_insert)

        for key, value in self.boolean_parameter_value_dqfd_map.items():
            value_to_insert = params.get(key)
            if value_to_insert:
                value.set(1)
            else:
                value.set(0)


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
