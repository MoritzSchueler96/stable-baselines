import pickle

import numpy as np

from stable_baselines.common.vec_env import VecEnvWrapper
from stable_baselines.common.running_mean_std import RunningMeanStd


class VecNormalize(VecEnvWrapper):
    """
    A moving average, normalizing wrapper for vectorized environment.
    has support for saving/loading moving average,

    :param venv: (VecEnv) the vectorized environment to wrap
    :param training: (bool) Whether to update or not the moving average
    :param norm_obs: (bool) Whether to normalize observation or not (default: True)
    :param norm_reward: (bool) Whether to normalize rewards or not (default: True)
    :param clip_obs: (float) Max absolute value for observation
    :param clip_reward: (float) Max value absolute for discounted reward
    :param gamma: (float) discount factor
    :param epsilon: (float) To avoid division by zero
    """

    def __init__(self, venv, training=True, norm_obs=True, norm_reward=True,
                 clip_obs=10., clip_reward=10., gamma=0.99, epsilon=1e-8, mean_mask=None):
        VecEnvWrapper.__init__(self, venv)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape, mean_mask=mean_mask)
        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        # Returns: discounted rewards
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.old_obs = np.array([])

    def step_wait(self):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)

        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        self.old_obs = obs
        obs = self._normalize_observation(obs)
        if self.norm_reward:
            if self.training:
                self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        self.ret[news] = 0
        return obs, rews, news, infos

    def _normalize_observation(self, obs):
        """
        :param obs: (numpy tensor)
        """
        if self.norm_obs:
            if self.training:
                self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs,
                          self.clip_obs)
            return obs
        else:
            return obs

    def get_original_obs(self):
        """
        returns the unnormalized observation

        :return: (numpy float)
        """
        return self.old_obs

    def reset(self, indices=None, *args, **kwargs):
        """
        Reset all environments
        """
        obs = self.venv.reset(indices, *args, **kwargs)
        if len(np.array(obs).shape) == 1:  # for when num_cpu is 1
            self.old_obs = [obs]
        else:
            self.old_obs = obs
        self.ret = np.zeros(self.num_envs)
        return self._normalize_observation(obs)

    def save_running_average(self, path, suffix=None):
        """
        :param path: (str) path to log dir
        :param suffix: (str) suffix to file
        """
        file_names = ['obs_rms', 'ret_rms']
        if suffix is not None:
            file_names = [f + suffix for f in file_names]
        for rms, name in zip([self.obs_rms, self.ret_rms], file_names):
            with open("{}/{}.pkl".format(path, name), 'wb') as file_handler:
                pickle.dump(rms, file_handler)

    def load_running_average(self, path, suffix=None):
        """
        :param path: (str) path to log dir
        :param suffix: (str) suffix to file
        """

        file_names = ['obs_rms', 'ret_rms']
        for name in file_names:
            open_name = name
            if suffix is not None:
                open_name += suffix
            with open("{}/{}.pkl".format(path, open_name), 'rb') as file_handler:
                setattr(self, name, pickle.load(file_handler))
