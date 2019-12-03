import random

import numpy as np

from stable_baselines.common.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    __name__ = "ReplayBuffer"

    def __init__(self, size):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

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

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

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

# TODO: add for dynamics randomization which also takes in my, and support for HER?
class RecurrentReplayBuffer(ReplayBuffer):
    __name__ = "RecurrentReplayBuffer"

    def __init__(self, size, episode_length, scan_length):
        super(RecurrentReplayBuffer, self).__init__(size)
        self._current_episode_data = []
        self._maxsize = self._maxsize // episode_length
        self._scan_length = scan_length
        #self._optim_length = optim_length # TODO: optim length

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        self._current_episode_data.append(data)
        if done:
            if self._next_idx >= len(self._storage):
                self._storage.append(self._current_episode_data)
            else:
                self._storage[self._next_idx] = self._current_episode_data
            self._next_idx = (self._next_idx + 1) % self._maxsize
            self._current_episode_data = []

    def __len__(self):
        return sum([len(episode) for episode in self._storage])

    def sample(self, batch_size, **_kwargs):
        num_episodes = len(self._storage)
        if num_episodes >= batch_size:
            ep_idxes = random.sample(range(num_episodes), k=batch_size)
        else:
            ep_idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        ep_ts = [random.randint(0, len(self._storage[ep_i])) for ep_i in ep_idxes]  # - self._optim_length)
        return self._encode_sample(ep_idxes, ep_ts)

    def _encode_sample(self, ep_idxes, ep_ts):
        obses_t, actions, rewards, obses_tp1, dones, hists = [], [], [], [], [], []
        for i in ep_idxes:
            ep_data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = ep_data[ep_ts[i]]
            ep_scan_start = ep_ts[i] - self._scan_length if ep_ts[i] - self._scan_length >= 0 else 0
            hist = [ep_data_h[0] for ep_data_h in ep_data[ep_scan_start:ep_ts[i]]]
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            hists.append(np.array(hist, copy=False))
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), hists


class EpisodicRecurrentReplayBuffer(ReplayBuffer):
    __name__ = "EpisodicRecurrentReplayBuffer"

    def __init__(self, size, episode_length):
        super(EpisodicRecurrentReplayBuffer, self).__init__(size // episode_length)
        self._episode_my = []
        self._current_episode_data = []

    def add(self, obs_t, action, reward, obs_tp1, done, goal, my=None):
        assert not (done and my is None)
        self._current_episode_data.append((obs_t, action, reward, obs_tp1, done, goal))

        if done:
            self.store_episode(my)

    def store_episode(self, my):
        if len(self._current_episode_data) == 0:
            return

        if self._next_idx >= len(self._storage):
            self._storage.append(self._current_episode_data)
            self._episode_my.append(my)
        else:
            self._storage[self._next_idx] = self._current_episode_data
            self._episode_my[self._next_idx] = my
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._current_episode_data = []

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones, goals, a_prevs, mys = [], [], [], [], [], [], [], []
        for i in idxes:
            ep_data = self._storage[i]
            for j, timestep in enumerate(ep_data):
                obs_t, action, reward, obs_tp1, done, goal = timestep
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
                goals.append(np.array(goal, copy=False))
                if j == 0:
                    a_prevs.append(np.zeros_like(ep_data[0][1]))
                else:
                    a_prevs.append(np.array(ep_data[j - 1][1], copy=False))
                mys.append(np.array(self._episode_my[i], copy=False))
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(goals), np.array(a_prevs), np.array(mys)

    def sample(self, batch_size, sample_max_len=-1):
        samples_left = batch_size
        obses_t, actions, rewards, obses_tp1, dones, goals, a_prevs, resets, mys = [], [], [], [], [], [], [], [], []
        while samples_left > 0:
            ep_idx = np.random.randint(0, len(self._storage) - 1)
            ep_data = self._storage[ep_idx]
            if sample_max_len != -1:
                ep_data = ep_data[np.random.randint(0, len(ep_data) - sample_max_len):]
            if len(ep_data) > samples_left:
                ep_data = ep_data[:samples_left]

            for j, timestep in enumerate(ep_data):
                obs_t, action, reward, obs_tp1, done, goal = timestep
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
                goals.append(np.array(goal, copy=False))
                if j == 0:
                    a_prevs.append(np.zeros_like(ep_data[0][1]))
                    resets.append(True)
                else:
                    resets.append(False)
                    a_prevs.append(np.array(ep_data[j - 1][1], copy=False))
                mys.append(np.array(self._episode_my[ep_idx], copy=False))

            samples_left -= len(ep_data)

            assert samples_left >= 0

            # TODO: rnn reset same as done?

        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(
            goals), np.array(a_prevs), np.array(resets), np.array(mys)

    def __len__(self):
        return sum([len(episode) for episode in self._storage])



class DRRecurrentReplayBuffer(RecurrentReplayBuffer):
    __name__ = "DRRecurrentReplayBuffer"

    def __init__(self, size, episode_length, scan_length, her_k=4):
        self.her_k = her_k
        super(DRRecurrentReplayBuffer, self).__init__(size, episode_length, scan_length)
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
                hist_o = obs_t
                hist_a = ep_data[ep_t - 1][1]
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
            hists_o.extend(hist_o)
            hists_a.extend(hist_a)
            goals.append(np.array(goal, copy=False))
            mys.append(np.array(self._episode_my[ep_i], copy=False))
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones), np.array(goals), np.array(hists_o), np.array(hists_a), np.array(mys)


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
        return tuple(list(encoded_sample) + [weights, idxes])

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

