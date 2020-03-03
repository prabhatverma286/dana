import json
import os

from baselines import deepq


def save_video_and_stats(episode_number):
    if episode_number == 1:
        return True
    return episode_number % 5 == 0 and episode_number is not 0


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


class DQNAgent(object):
    def __init__(self, output_dir, env, performance_identifier=None):
        self.action_space = env.action_space
        self.act = None
        self.output_dir = output_dir
        self.env = env
        if performance_identifier is not None:
            self.performance_identifier = performance_identifier
        else:
            self.performance_identifier = "Last Run"

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
        if self.env.episode_id > 1000:
            with open("rewards_buffer_size.json", "a+") as f:
                f.seek(0)
                contents = f.read()
                rewards = {}
                if contents is not None and (not contents.strip().__eq__("")):
                    rewards = json.loads(contents)

                rewards[self.performance_identifier] = lcl['episode_rewards'][:-1]
                f.seek(0)
                f.truncate()
                f.write(json.dumps(rewards))

            return True

        return False

    def train(self, env, arguments):
        self.act = deepq.learn(
            env,
            **arguments,
            callback=self._callback
        )
        self.env.close()

    def save(self, save_path):
        self.act.save(save_path)

    def from_path(self, arguments, load_path):
        arguments['total_timesteps'] = 0
        self.act = deepq.learn(self.env, **arguments, load_path=load_path)

    def take_action(self, observation):
        return self.act(observation[None])[0]
