import random

import numpy as np

from stable_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree

from sklearn.cluster import KMeans
from sklearn.exceptions import NotFittedError


class ReplayBuffer(object):
    __name__ = "ReplayBuffer"

    def __init__(self, size, extra_data_names=()):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0
        self._extra_data_names = sorted(extra_data_names)

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(np.ndarray, float, float, np.ndarray, bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done, *extra_data, bootstrap=None, **extra_data_kwargs):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        if bootstrap is not None:
            done = not bootstrap
        data = (obs_t, action, reward, obs_tp1, done, *extra_data,
                *[extra_data_kwargs[k] for k in sorted(extra_data_kwargs)])

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        extra_data = {name: [] for name in self._extra_data_names}
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, *extra_timestep_data = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)

            for data_i, extra_data_name in enumerate(self._extra_data_names):
                data = extra_timestep_data[data_i]
                if np.ndim(data) == 0:
                    extra_data[extra_data_name].append(data)
                else:
                    extra_data[extra_data_name].append(np.array(data, copy=False))

        extra_data = {k: np.array(v) for k, v in extra_data.items()}

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), extra_data

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class ClusteredReplayBuffer(ReplayBuffer):
    def __init__(self, size, cluster_on, n_clusters=5, recluster_every=0.25, strategy="single", fixed_idx=None, reducer=None, seed=None):
        super().__init__(size)
        data_idxs = {"obs": 0, "action": 1, "reward": 2, "obs_tp1": 3, "done": 4}
        if isinstance(cluster_on, list) or isinstance(cluster_on, tuple):
            self._cluster_on_idx = [data_idxs[data_name] for data_name in cluster_on]
        else:
            self._cluster_on_idx = [data_idxs[cluster_on]]
        self.cluster_alg = KMeans(n_clusters, random_state=seed)
        self._strategy = strategy
        self._fixed_idx = fixed_idx
        self._recluster_every = recluster_every  # TODO: should resample dynamically less and less
        self._samples_until_recluster = 10000
        self._cluster_sample_idxs = [[] for i in range(n_clusters)]
        self._n_clusters = n_clusters
        self.reducer = reducer

    def add(self, obs_t, action, reward, obs_tp1, done, *extra_data, bootstrap=None, **extra_data_kwargs):
        if bootstrap is not None:
            done = not bootstrap
            
        cluster_data = []
        for cluster_data_idx in self._cluster_on_idx:
            if cluster_data_idx == 0:
                c_d = obs_t
            elif cluster_data_idx == 1:
                c_d = action
            elif cluster_data_idx == 2:
                c_d = reward
            elif cluster_data_idx == 3:
                c_d = obs_tp1,
            elif cluster_data_idx == 4:
                c_d = done
            else:
                raise ValueError
            cluster_data.append(c_d)
        cluster_data = np.concatenate(cluster_data)[None]
        try:
            cluster_idx = self.cluster_alg.predict(cluster_data)[0]
        except NotFittedError:
            cluster_idx = 0

        data = [obs_t, action, reward, obs_tp1, done, *extra_data, *[extra_data_kwargs[k] for k in sorted(extra_data_kwargs)]]
        data.append(cluster_idx)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            self._cluster_sample_idxs[cluster_idx].append(self._next_idx)
        else:
            self._storage[self._next_idx] = data
            self._cluster_sample_idxs[self._storage[self._next_idx][-1]].remove(self._next_idx)
            self._cluster_sample_idxs[cluster_idx].append(self._next_idx)
        self._next_idx = (self._next_idx + 1) % self._maxsize

        self._samples_until_recluster -= 1
        if self._samples_until_recluster <= 0:
            self.fit_and_assign_clusters()
            self._samples_until_recluster = int(max(20000, len(self) * self._recluster_every))

    def batch_add(self, batch):
        cluster_idxs = self.cluster_alg.predict([np.concatenate([np.atleast_1d(s[idx]) for idx in self._cluster_on_idx]) for s in batch])
        for i, idx in enumerate(cluster_idxs):
            self._storage.append([*batch[i], idx])
            self._cluster_sample_idxs[idx].append(i)

    def sample(self, batch_size, strategy=None):
        if strategy is None:
            strategy = self._strategy
        if strategy == "single":
            cluster_sample_count = np.array([len(self._cluster_sample_idxs[c_i]) for c_i in range(self._n_clusters)])
            cluster_idx = np.random.choice(range(self._n_clusters), p=cluster_sample_count / np.sum(cluster_sample_count))
            sample_idxs = [self._cluster_sample_idxs[cluster_idx][random.randint(0, len(self._cluster_sample_idxs[cluster_idx]) - 1)] for _ in range(batch_size)]
            return super()._encode_sample(sample_idxs)
            sample_idxs = [self._cluster_sample_idxs[cluster_idx][random.randint(0, len(self._cluster_sample_idxs[cluster_idx]) - 1)] for _ in range(batch_size)]
            return super()._encode_sample(sample_idxs)
        elif strategy in ["uniform", "proportional"]:
            if strategy == "uniform":
                samples_per_cluster = [batch_size // self._n_clusters] * self._n_clusters
                num_samples_left_over = batch_size % self._n_clusters
                weights = None
            elif strategy == "proportional":
                total_size = sum([len(c_s_i) for c_s_i in self._cluster_sample_idxs])
                samples_per_cluster = np.array([max(int(len(self._cluster_sample_idxs[i]) / total_size * batch_size), 1) for i in range(self._n_clusters)])
                num_samples_left_over = batch_size - np.sum(samples_per_cluster)
                if num_samples_left_over < 0:
                    highest_sample_counts_idxs = np.argpartition(samples_per_cluster, num_samples_left_over)[num_samples_left_over:]
                    samples_per_cluster[highest_sample_counts_idxs] -= 1
                    num_samples_left_over = 0
                weights = samples_per_cluster

            leftover_samples_cluster_idxs = random.choices(range(len(self._cluster_sample_idxs)), k=num_samples_left_over,
                                                           weights=weights)
            for idx in leftover_samples_cluster_idxs:
                samples_per_cluster[idx] += 1

            data = [[] for i in range(5)]
            for cluster_i in range(self._n_clusters):
                sample_idxs = [self._cluster_sample_idxs[cluster_i][random.randint(0, len(self._cluster_sample_idxs[cluster_i]) - 1)]
                               for _ in range(samples_per_cluster[cluster_i])]
                c_s = super()._encode_sample(sample_idxs)
                for i in range(5):
                    data[i].append(c_s[i])

            res = []
            for i in range(5):
                res.append(np.concatenate(data[i], axis=0))

            return res[0], res[1], res[2], res[3], res[4]
        else:
            raise ValueError # TODO: even/uniform/distribution

    def fit_and_assign_clusters(self):
        sample_cluster_idxs = self.cluster_alg.fit_predict([np.concatenate([s[c_i] for c_i in self._cluster_on_idx]) for s in self._storage])
        self._cluster_sample_idxs = [[] for i in range(len(self._cluster_sample_idxs))]

        for s_i in range(len(self._storage)):
            self._cluster_sample_idxs[sample_cluster_idxs[s_i]].append(s_i)
            self._storage[s_i][-1] = sample_cluster_idxs[s_i]

    def can_sample(self, n_samples):  # TODO: adjust for strategy etc.
        return not any([len(c_s_i) < n_samples for c_s_i in self._cluster_sample_idxs])


# TODO: lots of work to do on this one
class RecurrentReplayBuffer(ReplayBuffer):
    __name__ = "RecurrentReplayBuffer"

    def __init__(self, size, episode_length, scan_length, rnn_inputs=(), extra_data_names=(), her_k=4):
        super().__init__(size)
        self._maxsize = self._maxsize // episode_length
        self.her_k = her_k
        self.scan_length = scan_length
        self._extra_data_names = extra_data_names
        self._rnn_inputs = rnn_inputs
        self._data_idxs = {"obs": 0, "action": 1, "reward": 2, "obs_tp1": 3, "done": 4,
                           **{name: 5 + i for i, name in enumerate(self._extra_data_names)}}
        self._rnn_data_idxs = [self._data_idxs[name] for name in self._rnn_inputs]
        self._current_episode_data = []

    def add(self, obs_t, action, reward, obs_tp1, done, *extra_data):
        if self.her_k > 0:
            obs_t = [obs_t]
            obs_tp1 = [obs_tp1]
            reward = [reward]
        data = (obs_t, action, reward, obs_tp1, done, *extra_data)
        self._current_episode_data.append(data)
        if done:
            if self._next_idx >= len(self._storage):
                self._storage.append(self._current_episode_data)
            else:
                self._storage[self._next_idx] = self._current_episode_data
            self._next_idx = (self._next_idx + 1) % self._maxsize
            self._current_episode_data = []

    def add_her(self, obs, obs_tp1, reward, timestep, ep_index=None):
        assert self.her_k > 0
        if ep_index is not None:
            episode_data = self._storage[ep_index]
        else:
            episode_data = self._current_episode_data
        episode_data[timestep][0].append(obs)
        episode_data[timestep][2].append(reward)
        episode_data[timestep][3].append(obs_tp1)

    def sample(self, batch_size, **_kwargs):
        if self.her_k > 0:
            num_episodes = len(self._storage)
            if num_episodes >= batch_size:
                ep_idxes = random.sample(range(num_episodes), k=batch_size)
            else:
                ep_idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
            ep_ts = [random.randint(self.scan_length * (1 + self.her_k), (len(self._storage[ep_i]) - 1) * (1 + self.her_k)) for ep_i in ep_idxes]  # - self._optim_length)
            return self._encode_sample(ep_idxes, ep_ts)
        else:
            return super().sample(batch_size)

    def _encode_sample(self, ep_idxes, ep_ts):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        extra_data_not_rnn = [name for name in self._extra_data_names if name not in self._rnn_inputs]
        extra_data_not_rnn_idxs = [self._data_idxs[name] for name in extra_data_not_rnn]
        extra_data = [[] for i in range(len(extra_data_not_rnn))]
        hists = [[] for i in range(len(self._rnn_inputs))]

        for i, ep_i in enumerate(ep_idxes):
            if self.her_k > 0:
                ep_t = int(ep_ts[i] / (self.her_k + 1))
            else:
                ep_t = ep_ts[i]
            ep_data = self._storage[ep_i]
            obs_t, action, reward, obs_tp1, done, *extra_timestep_data = ep_data[ep_t]
            if self.her_k > 0:
                her_idx = ep_ts[i] - ep_t * (self.her_k + 1)
                obs_t, obs_tp1, reward = obs_t[her_idx], obs_tp1[her_idx], reward[her_idx]
            if self.scan_length > 0:
                ep_scan_start = ep_t - self.scan_length if ep_t - self.scan_length >= 0 else 0
                ep_hists = [[] for i in range(len(self._rnn_inputs))]
                for hist_i in range(ep_scan_start, ep_t + 1):
                    for input_i, data_idx in enumerate(self._rnn_data_idxs):
                        hist_data = ep_data[hist_i][data_idx]
                        if self.her_k > 0 and data_idx in [0, 2, 3]:
                            hist_data = hist_data[ep_ts[i] - ep_t * (self.her_k + 1)]
                        ep_hists[input_i].append(np.array(hist_data))
            else:
                ep_hists = [[ep_data[data_idx]] for data_idx in self._rnn_data_idxs]
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            for hist_i in range(len(hists)):
                hists[hist_i].extend(ep_hists[hist_i])
            for j, data_i in enumerate(extra_data_not_rnn_idxs):
                if np.ndim(extra_timestep_data[j]) == 0:
                    extra_data[j].append(extra_timestep_data[data_i - 5])
                else:
                    extra_data[j].append(np.array(extra_timestep_data[data_i - 5], copy=False))

        extra_data_dict = {name: np.array(extra_data[i]) for i, name in enumerate(extra_data_not_rnn)}
        batch_resets = np.zeros(shape=(len(ep_idxes) * (self.scan_length + 1)), dtype=np.bool)
        batch_resets[::self.scan_length] = 1
        extra_data_dict["reset"] = batch_resets
        res = [np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones),
            extra_data_dict]
        for i, name in enumerate(self._rnn_inputs):
            rnn_input_idx = self._rnn_data_idxs[i]
            if rnn_input_idx < 5:
                res[rnn_input_idx] = np.array(hists[i])
            else:
                res[-1][name] = np.array(hists[i])


        return res

    def __len__(self):
        return sum([max(0, len(ep) - self.scan_length) for ep in self._storage])


# TODO: maybe add support for episode constant data
class EpisodicRecurrentReplayBuffer(ReplayBuffer):
    __name__ = "EpisodicRecurrentReplayBuffer"

    def __init__(self, size, episode_length, sample_consecutive_max=-1, extra_data_names=()):
        super().__init__(size // episode_length)
        self._current_episode_data = []
        #self._episode_data = []  # Data which is constant within episode
        self._extra_data_names = sorted(extra_data_names)
        self._sample_consecutive_max = sample_consecutive_max

    def add(self, obs_t, action, reward, obs_tp1, done, *extra_data):
        self._current_episode_data.append((obs_t, action, reward, obs_tp1, done, *extra_data))

        if done:
            self.store_episode()

    def store_episode(self):
        if len(self._current_episode_data) == 0:
            return

        if self._next_idx >= len(self._storage):
            self._storage.append(self._current_episode_data)
        else:
            self._storage[self._next_idx] = self._current_episode_data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._current_episode_data = []

    def sample(self, batch_size, sample_consecutive_max=None):
        if sample_consecutive_max is None:
            sample_consecutive_max = self._sample_consecutive_max
        samples_left = batch_size
        obses_t, actions, rewards, obses_tp1, dones, resets = [], [], [], [], [], []
        extra_data = [[] for i in range(len(self._extra_data_names))]
        while samples_left > 0:
            ep_idx = np.random.randint(0, len(self._storage) - 1)
            ep_data = self._storage[ep_idx]
            if sample_consecutive_max != -1:
                ep_start_idx = np.random.randint(0, max(len(ep_data) - sample_consecutive_max + 1, 1))
                ep_data = ep_data[ep_start_idx:ep_start_idx + sample_consecutive_max]
            if len(ep_data) > samples_left:
                ep_data = ep_data[:samples_left]

            for j, timestep in enumerate(ep_data):
                obs_t, action, reward, obs_tp1, done, *extra_timestep_data = timestep
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
                resets.append(True if j == 0 else False)
                for data_i, data in enumerate(extra_timestep_data):
                    if np.ndim(data) == 0:
                        extra_data[data_i].append(data)
                    else:
                        extra_data[data_i].append(np.array(data, copy=False))

            samples_left -= len(ep_data)

            assert samples_left >= 0

        extra_data_dict = {name: np.array(extra_data[i]) for i, name in enumerate(self._extra_data_names)}
        extra_data_dict["reset"] = np.array(resets)

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), extra_data_dict

    def __len__(self):
        if len(self.storage) > 1:
            return sum([len(episode) for episode in self._storage])
        else:
            return 0


class DRRecurrentReplayBuffer(ReplayBuffer):
    __name__ = "DRRecurrentReplayBuffer"

    def __init__(self, size, episode_max_len, scan_length, her_k=4):
        self.her_k = her_k
        super().__init__(size)
        self._scan_length = scan_length
        self._maxsize = self._maxsize // episode_max_len
        self._episode_my = []

    def add(self, obs_t, action, reward, obs_tp1, done, goal, my=None):
        assert not (done and my is None)

        if self.her_k > 0:
            goal = [goal]
            reward = [reward]
        data = (obs_t, action, reward, obs_tp1, done, goal)
        self._current_episode_data.append(data)
        if done:
            if self._next_idx >= len(self._storage):
                self._storage.append(self._current_episode_data)
                self._episode_my.append(my)
            else:
                self._storage[self._next_idx] = self._current_episode_data
                self._episode_my[self._next_idx] = my
            self._next_idx = (self._next_idx + 1) % self._maxsize
            self._current_episode_data = []

    def add_her(self, goal, reward, timestep, ep_index=None):
        assert self.her_k > 0
        if ep_index is not None:
            episode_data = self._storage[ep_index]
        else:
            episode_data = self._current_episode_data
        episode_data[timestep][5].append(goal)
        episode_data[timestep][2].append(reward)

    def sample(self, batch_size, **_kwargs):
        if self.her_k > 0:
            num_episodes = len(self._storage)
            if num_episodes >= batch_size:
                ep_idxes = random.sample(range(num_episodes), k=batch_size)
            else:
                ep_idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
            ep_ts = [random.randint(self._scan_length * (1 + self.her_k), (len(self._storage[ep_i]) - 1) * (1 + self.her_k)) for ep_i in ep_idxes]  # - self._optim_length)
            return self._encode_sample(ep_idxes, ep_ts)
        else:
            return super(DRRecurrentReplayBuffer, self).sample(batch_size)

    def _encode_sample(self, ep_idxes, ep_ts):
        obses_t, actions, rewards, obses_tp1, dones, goals, mys, hists_o, hists_a = [], [], [], [], [], [], [], [], []
        for i, ep_i in enumerate(ep_idxes):
            if self.her_k > 0:
                ep_t = int(ep_ts[i] / (self.her_k + 1))
            else:
                ep_t = ep_ts[i]
            ep_data = self._storage[ep_i]
            obs_t, action, reward, obs_tp1, done, goal = ep_data[ep_t]
            if self.her_k > 0:
                goal = goal[ep_ts[i] - ep_t * (self.her_k + 1)]
                reward = reward[ep_ts[i] - ep_t * (self.her_k + 1)]
            if self._scan_length > 0:
                ep_scan_start = ep_t - self._scan_length if ep_t - self._scan_length >= 0 else 0
                hist_o, hist_a = [], []
                for hist_i in range(ep_scan_start, ep_t):
                    hist_o.append(np.array(ep_data[hist_i][0]))
                    if hist_i > 0:
                        hist_a.append(np.array(ep_data[hist_i - 1][1]))
                    else:
                        hist_a.append(np.zeros(shape=(len(ep_data[0][1]),)))
                hist_o.append(np.array(obs_t))
                hist_a.append(np.array(ep_data[ep_t - 1][1]))
            else:
                hist_o = [obs_t]
                hist_a = [ep_data[ep_t - 1][1]]
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            hists_o.extend(hist_o)
            hists_a.extend(hist_a)
            goals.append(np.array(goal, copy=False))
            mys.append(np.array(self._episode_my[ep_i], copy=False))
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), {"goal": np.array(goals), "obs_rnn": np.array(hists_o), "action_prev": np.array(hists_a), "my": np.array(mys)}


class PrioritizedReplayBuffer(ReplayBuffer):
    __name__ = "PrioritizedReplayBuffer"

    def __init__(self, size, alpha):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            mass = random.random() * self._it_sum.sum(0, len(self._storage) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta=0):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        #return tuple(list(encoded_sample) + [weights, idxes])
        return encoded_sample[0], encoded_sample[1], encoded_sample[2], encoded_sample[3], encoded_sample[4],  {"is_weights": weights, "idxs": idxes}

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        priorities += 1e-8
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class DiscrepancyReplayBuffer(ReplayBuffer):
    def __init__(self, size, scorer):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(DiscrepancyReplayBuffer, self).__init__(size)
        self.scores = []
        self.scorer = scorer
        self.min_score = None
        self.max_score = None

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx

        score = self.scorer(np.expand_dims(obs_tp1, axis=0))[0][0]
        if self.min_score is None or score < self.min_score:
            self.min_score = score
        if self.max_score is None or score > self.max_score:
            self.max_score = score
        if self._next_idx >= len(self._storage):
            self.scores.append(score)
        else:
            self.scores[idx] = score

        super().add(obs_t, action, reward, obs_tp1, done)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        if not self.can_sample(batch_size):
            return self._encode_sample(list(range(len(self))))

        scores = self._scale_scores(np.array(self.scores))
        idxs = np.random.choice(np.arange(len(scores)), size=(batch_size,), p=scores / np.sum(scores), replace=False)

        return self._encode_sample(idxs)

    def update_priorities(self):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        scores = self.scorer([transition[0] for transition in self.storage])[:, 0]
        for i in range(len(self)):
            self.scores[i] = scores[i]

    def _scale_scores(self, vals):
        return (vals - self.min_score) / (self.max_score - self.min_score) * (1 - 0.1) + 0.1


class StableReplayBuffer(ReplayBuffer):
    __name__ = "StableReplayBuffer"

    def __init__(self, size):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(StableReplayBuffer, self).__init__(size)
        self.scores = []
        self.lower_clip = None
        self.upper_clip = None

    def add(self, obs_t, action, reward, obs_tp1, done, score=None):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx

        if self._next_idx >= len(self._storage):
            self.scores.append(score)
        else:
            self.scores[idx] = score

        super().add(obs_t, action, reward, obs_tp1, done)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        if not self.can_sample(batch_size):
            return self._encode_sample(list(range(len(self))))

        scores = np.array(self.scores)
        scores = np.clip(scores, np.percentile(scores, 10), np.percentile(scores, 90))
        idxs = np.random.choice(np.arange(len(self.scores)), size=(batch_size,), p=scores / np.sum(scores), replace=False)

        return self._encode_sample(idxs)

    def update_priorities(self):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        self.lower_clip = np.percentile(self.scores, 10)
        self.upper_clip = np.percentile(self.scores, 90)

