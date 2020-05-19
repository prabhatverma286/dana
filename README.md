# DQN Agent for Atari
This repository provides the code for a DQN agent that learns how to play certain video games. The agent is named Dqn AgeNt for Atari (DANA) - and currently supports the game of Cartpole, Pong and Breakout. The code provides a demo system to view the progress of the agent and tune the hyperparameters for the training process of the agent. With the help of appropriate ROMS (described below), the agent can also be used to solve the Sonic the Hedgehog - Genesis environment.

The demo system also allows the user to launch the [Deep Q Learning from Demonstrations](https://arxiv.org/abs/1704.03732) algorithm - which involves separate pre-training and training phases. 

The demo system makes use of the [Open AI Baselines](https://github.com/openai/baselines) library to implement the DQN algorithm, and [this DQfD implementation](https://github.com/AurelianTactics/dqfd-with-keras).

This project was completed as my third year project at The University of Manchester.

The project was developed and tested on Windows 10. 

## Installation
The author recommends the use of anaconda environments to manage different python versions, and using pip to install the python packages.
1. Checkout the source code from https://github.com/prabhatverma286/dana using git (Follow instructions on https://git-scm.com/download/win to install git)
    >git clone https://github.com/prabhatverma286/dana.git
2. Install Anaconda from https://docs.anaconda.com/anaconda/install/
3. Create a new environment called 'dana' with python 3.5
    > conda create --name dana python=3.5
4. Activate the environment
    > conda activate dana
5. Make sure that pip is installed
    >conda install pip
6. Navigate to the project directory. Use pip to install the requirements listed in requirements.txt
    >pip install -r requirements.txt
7. Package atari-py needs a specific version to work with windows. Use the following command.
    >pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
8. Navigate to the base directory. Install OpenAI Gym
    ```
    git clone https://github.com/openai/gym.git
    cd gym
    pip install -e .
    ```
9. Navigate to the base directory. Install OpenAI Baselines
    ```
    git clone https://github.com/openai/baselines.git
    cd baselines 
    pip install -e .
    ```
 10. Navigate to the base directory. Install the retro-contest
      ```
      git clone --recursive https://github.com/openai/retro-contest.git
      cd retro-contest
      pip install -e "support[docker,rest]"
      ```
 11. Since Sonic the Hedgehog - Genesis is copyrighted by SEGA, we first need to buy and install the game from https://store.steampowered.com/app/71113/Sonic_The_Hedgehog/
 12. Run the following command (the argument would change according to where your Steam client is installed)
      >python -m retro.import "C:\Program Files (x86)\Steam\steamapps\common\Sega Classics\uncompressed ROMs"
13. Navigate to the project (DANA) directory and run the demo system using ```python .```      
    
## Usage

Using the demo system, one can train the agent on the games of Cartpole, Pong, Breakout and Sonic. The repository also contains some pre-trained models for the said games, which can be run in the "evaluation" mode to see the agent play these games with superhuman efficiency. The models for Sonic are suboptimal as they were not trained for enough timesteps due to time contraints of the project.

The DQfD implementation supports the pre-training, training and evaluation phases for the agent. There is no preview available for the pre-training phase, as the agent does not interact with the environment in this phase. Human demonstrations are also included in the repository for the DQfD algorithm. These demos were taken from https://github.com/openai/retro-movies.

The training process for the agent can be fine-tuned using different parameters. These parameters plug in directly to the OpenAI Baselines library. Since the development of the project, a new repository by OpenAI called stable-baselines has been released. The parameters used are exactly the same and hence, their explanation and documentation can be found at https://stable-baselines.readthedocs.io/en/master/modules/dqn.html.

The demo system saves the videos of the agent in the agent-evaluation or the agent-training directory. These are overwritten with each new training/evaluation.  

The train.py and evaluate.py files provide an easy way of running the agent without the overhead of displaying videos using the demo system.