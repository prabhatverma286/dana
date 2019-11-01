import json
import os

from baselines import deepq


def save_video_and_stats(episode_number):
    return episode_number % 25 == 0 and episode_number is not 0


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class DQNAgent(object):
    def __init__(self, output_dir, env):
        self.action_space = env.action_space
        self.act = None
        self.output_dir = output_dir
        self.env = env

    def _callback(self, lcl, _glb):
        if save_video_and_stats(self.env.episode_id):
            # write stats in json
            file_name = os.path.join(self.output_dir, str(self.env.episode_id) + '-meta.json')
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, 'w+') as fd:
                fd.write(json.dumps({
                    'episode_number': self.env.episode_id,
                    'episode_score': lcl['episode_rewards'][-1]
                }))

        # stop training if reward exceeds 199
        is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
        return is_solved

    def train(self):
        self.act = deepq.learn(
            self.env,
            network='mlp',
            lr=1e-3,
            total_timesteps=100000,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            print_freq=10,
            callback=self._callback
        )

    def save(self, save_path):
        self.act.save(save_path)

    def from_path(self, load_path):
        self.act = deepq.learn(self.env, network='mlp', total_timesteps=0, load_path=load_path)

    def take_action(self, observation):
        return self.act(observation[None])[0]
