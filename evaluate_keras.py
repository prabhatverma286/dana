import json
import os
import keras.backend as K
import numpy as np
import tensorflow as tf

import gym
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind, WarpFrame, FrameStack
from baselines.common.retro_wrappers import make_retro
from baselines import logger
import argparse

from gym.wrappers import Monitor
from keras import Input, initializers, regularizers, Model
from keras.layers import Lambda, Conv2D, Flatten, Dense
from keras.optimizers import Adam

from Helpers import BreakoutMonitor, envs
from agent import DQNAgent


def save_video_and_stats(episode_number):
    if episode_number == 1:
        return True
    # return episode_number % 5 == 0 and episode_number is not 0
    return True


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env, action_list):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        # actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B'], ['B']]
        #actions = [['LEFT'], ['RIGHT'], ['LEFT', 'B'], ['RIGHT', 'B'],  ['NOOP'],['B']]
        #actions = [['LEFT'], ['RIGHT'], ['DOWN'], ['NOOP'], ['B']]
        #actions = [['LEFT'], ['RIGHT'], ['B']]
        #actions = [['NOOP'], ['LEFT'], ['RIGHT'], ['LEFT', 'B'], ['RIGHT', 'B'], ['DOWN'],['B']]
        actions = action_list

        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            if action == ['NOOP']:
                self._actions.append(arr)
                continue
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


def main(env_id, arguments):
    logger.set_level(logger.INFO)

    env = make_retro(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', scenario='contest')
    env = bench.Monitor(env, logger.get_dir())
    # env = deepq.wrap_atari_dqn(env)

    action_list = [['NOOP'], ['LEFT'], ['RIGHT'], ['LEFT', 'B'], ['RIGHT', 'B'], ['DOWN'], ['B']]

    env = SonicDiscretizer(env, action_list)
    env = AllowBacktracking(env)
    # env = RewardScaler(env)

    env = WarpFrame(env)
    env = FrameStack(env, 4)

    # if env_id == envs["Cartpole"]:
    #     env = gym.make(env_id)
    # else:
    #     env = make_atari(env_id)
    #     env = wrap_deepmind(env, frame_stack=True, scale=False, clip_rewards=False)

    out_dir = 'agent-evaluation'
    os.makedirs(out_dir, exist_ok=True)

    if env_id == envs["Breakout"]:
        env = BreakoutMonitor(env, directory=out_dir, force=True, video_callable=save_video_and_stats)
    else:
        env = Monitor(env, directory=out_dir, force=True, video_callable=save_video_and_stats)

    env.seed(0)

    model = build_model(7)
    model.load_weights("saved_models\\SonicTheHedgehog_model.pkl")
    agent = DQNAgent(out_dir, env)
    # agent.from_path(arguments, "saved_models\\" + env_id + "_model.pkl")

    empty_by_one = np.zeros((1, 1))
    empty_exp_action_by_one = np.zeros((1, 2))
    empty_action_len_by_one = np.zeros((1, 7))

    for i in range(100):
        ob = env.reset()
        total_reward = 0
        while True:
            temp_curr_obs = np.array(ob)
            temp_curr_obs = temp_curr_obs.reshape(1, temp_curr_obs.shape[0], temp_curr_obs.shape[1],
                                                  temp_curr_obs.shape[2])
            action, _, _ = model.predict([temp_curr_obs, temp_curr_obs,empty_by_one, empty_exp_action_by_one,empty_action_len_by_one])
            action = np.argmax(action)
            ob, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward
            if done:
                file_name = os.path.join(out_dir, str(i + 1) + '-meta.json')
                with open(file_name, 'w') as fd:
                    fd.write(json.dumps({
                        'episode_number': i + 1,
                        'episode_score': total_reward
                    }))
                print("reward: {}".format(total_reward))
                break
    env.close()


def build_model(action_len, img_rows=84, img_cols=84, img_channels=4, dueling=True, clip_value=1.0,
                learning_rate=1e-4, nstep_reg=1.0, slmc_reg=1.0, l2_reg=10e-5):

    input_img = Input(shape=(img_rows, img_cols, img_channels), name='input_img', dtype='float32')
    scale_img = Lambda(lambda x: x/255.)(input_img) #scales the image. input is in ints of 0 to 255
    layer_1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4), padding='same',
                     activation='relu', input_shape=(img_rows, img_cols, img_channels),
                     kernel_initializer=initializers.glorot_normal(seed=31),
                     kernel_regularizer=regularizers.l2(l2_reg),
                     bias_regularizer=regularizers.l2(l2_reg))(scale_img)#(input_img)
    layer_2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu',
                     kernel_initializer=initializers.glorot_normal(seed=31),
                     kernel_regularizer=regularizers.l2(l2_reg),
                     bias_regularizer=regularizers.l2(l2_reg))(layer_1)
    layer_3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer=initializers.glorot_normal(seed=31),
                     kernel_regularizer=regularizers.l2(l2_reg),
                     bias_regularizer=regularizers.l2(l2_reg))(layer_2)
    x = Flatten()(layer_3)
    x = Dense(256, activation='relu',
              kernel_initializer=initializers.glorot_normal(seed=31),
              kernel_regularizer=regularizers.l2(l2_reg),
              bias_regularizer=regularizers.l2(l2_reg))(x)
    if not dueling:
        cnn_output = Dense(action_len,
                           kernel_initializer=initializers.glorot_normal(seed=31),
                           kernel_regularizer=regularizers.l2(l2_reg),
                           bias_regularizer=regularizers.l2(l2_reg), name='cnn_output')(x)
    else:
        dueling_values = Dense(1,
                               kernel_initializer=initializers.glorot_normal(seed=31),
                               kernel_regularizer=regularizers.l2(l2_reg),
                               bias_regularizer=regularizers.l2(l2_reg), name='dueling_values')(x)
        dueling_actions = Dense(action_len,
                                kernel_initializer=initializers.glorot_normal(seed=31),
                                kernel_regularizer=regularizers.l2(l2_reg),
                                bias_regularizer=regularizers.l2(l2_reg), name='dq_actions')(x)

        # https://github.com/keras-team/keras/issues/2364
        def dueling_operator(duel_input):
            duel_v = duel_input[0]
            duel_a = duel_input[1]
            return duel_v + (duel_a - K.mean(duel_a, axis=1, keepdims=True))

        cnn_output = Lambda(dueling_operator, name='cnn_output')([dueling_values, dueling_actions])
        # alternate way: https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py
    cnn_model = Model(input_img, cnn_output)

    input_img_dq = Input(shape=(img_rows, img_cols, img_channels), name='input_img_dq', dtype='float32')
    input_img_nstep = Input(shape=(img_rows, img_cols, img_channels), name='input_img_nstep', dtype='float32')
    dq_output = cnn_model(input_img_dq)
    nstep_output = cnn_model(input_img_nstep)

    # supervised large margin classifier loss
    # max[Q(s,a)+l(ae,a)] - Q(s,ae) if expert replay, 0 otherwise
    # minimize it with mean absolute error vs. fake target values of 0
    input_is_expert = Input(shape=(1,), name='input_is_expert')
    input_expert_action = Input(shape=(2,), name='input_expert_action', dtype='int32')
    input_expert_margin = Input(shape=(action_len,), name='input_expert_margin')

    def slmc_operator(slmc_input):
        is_exp = slmc_input[0]
        sa_values = slmc_input[1]
        exp_act = K.cast(slmc_input[2], dtype='int32')
        exp_margin = slmc_input[3]

        #exp_val = tf.gather(sa_values, exp_act, axis=-1)
        exp_val = tf.gather_nd(sa_values, exp_act)
        # not sure how to use arange with Keras like you can with numpy so convert to numpy then back to Keras
        # I want state,action values along the expert action choice. easy with np.arange, so do that part outside of model
        #         exp_identity = K.ones(shape=K.shape(sa_values)) * expert_margin
        #         exp_identity[K.arange(sa_len),K.eval(exp_act)] = 0.
        #         exp_val = sa_values[K.arange(sa_len),exp_act]


        max_margin = K.max(sa_values + exp_margin, axis=1)
        max_margin_2 = max_margin - exp_val
        max_margin_3 = K.reshape(max_margin_2,K.shape(is_exp))
        max_margin_4 = tf.multiply(is_exp,max_margin_3)
        return max_margin_4


    slmc_output = Lambda(slmc_operator, name='slmc_output')([input_is_expert, dq_output,
                                                             input_expert_action, input_expert_margin])

    model = Model(inputs=[input_img_dq, input_img_nstep, input_is_expert, input_expert_action, input_expert_margin],
                  outputs=[dq_output, nstep_output, slmc_output])

    if clip_value is not None:
        adam = Adam(lr=learning_rate, clipvalue=clip_value)
    else:
        adam = Adam(lr=learning_rate)

    model.compile(optimizer=adam,
                  loss=['mse', 'mse', 'mae'],
                  loss_weights=[1., nstep_reg, slmc_reg])

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', help='gym env id')
    parser.add_argument('--json_arguments', type=json.loads, help='gym env id')

    args = parser.parse_args()
    main(args.env_id, args.json_arguments)
