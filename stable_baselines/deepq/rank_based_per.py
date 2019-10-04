import sys
import math
import random
import numpy as np
from .binary_heap import BinaryHeap


class RankPrioritizedReplayBuffer(object):
    __name__ = "RankPrioritizedReplayBuffer"

    def __init__(self, size, alpha, learning_starts, batch_size=32):
        self.size = size
        self.replace_flag = True
        self.priority_size = self.size

        self.alpha = alpha
        # partition number N, split total size to N part
        self.partition_num = batch_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size

        self.index = 0
        self.record_size = 0
        self.isFull = False

        self._storage = {}
        self.priority_queue = BinaryHeap(self.priority_size)
        self.distributions = self.build_distributions()

    def build_distributions(self):
        """
        preprocess pow of rank
        (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
        :return: distributions, dict
        """
        res = {}
        n_partitions = self.partition_num
        partition_num = 1
        # each part size
        partition_size = int(math.floor(self.size / n_partitions))

        for n in range(partition_size, self.size + 1, partition_size):
            if self.learning_starts <= n <= self.priority_size:
                distribution = {}
                # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
                pdf = list(
                    map(lambda x: math.pow(x, -self.alpha), range(1, n + 1))
                )
                pdf_sum = math.fsum(pdf)
                distribution['pdf'] = list(map(lambda x: x / pdf_sum, pdf))
                # split to k segment, and than uniform sample in each k
                # set k = batch_size, each segment has total probability is 1 / batch_size
                # strata_ends keep each segment start pos and end pos
                cdf = np.cumsum(distribution['pdf'])
                strata_ends = {1: 0, self.batch_size + 1: n}
                step = 1 / float(self.batch_size)
                index = 1
                for s in range(2, self.batch_size + 1):
                    while cdf[index] < step:
                        index += 1
                    strata_ends[s] = index
                    step += 1 / float(self.batch_size)

                distribution['strata_ends'] = strata_ends

                res[partition_num] = distribution

            partition_num += 1

        return res

    def fix_index(self):
        """
        get next insert index
        :return: index, int
        """
        if self.record_size <= self.size:
            self.record_size += 1
        if self.index % self.size == 0:
            self.isFull = True if len(self._storage) == self.size else False
            if self.replace_flag:
                self.index = 1
                return self.index
            else:
                sys.stderr.write('Experience replay buff is full and replace is set to FALSE!\n')
                return -1
        else:
            self.index += 1
            return self.index

    def can_sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        return self.record_size >= batch_size

    def add(self, obs, action, reward, new_obs, done):
        """
        store experience, suggest that experience is a tuple of (s1, a, r, s2, t)
        so each experience is valid
        :param experience: maybe a tuple, or list
        :return: bool, indicate insert status
        """
        experience = (obs, action, reward, new_obs, done)
        insert_index = self.fix_index()
        if insert_index > 0:
            if insert_index in self._storage:
                del self._storage[insert_index]
            self._storage[insert_index] = experience
            # add to priority queue
            priority = self.priority_queue.get_max_priority()
            self.priority_queue.update(priority, insert_index)
            return True
        else:
            sys.stderr.write('Insert failed\n')
            return False

    def retrieve(self, indices):
        """
        get experience from indices
        :param indices: list of experience id
        :return: experience replay sample
        """
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in indices:
            if i == 0:
                i = 1
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def rebalance(self):
        """
        rebalance priority queue
        :return: None
        """
        self.priority_queue.balance_tree()

    def update_priorities(self, indices, delta):
        """
        update priority according indices and deltas
        :param indices: list of experience id
        :param delta: list of delta, order correspond to indices
        :return: None
        """
        for i in range(0, len(indices)):
            self.priority_queue.update(math.fabs(delta[i]), indices[i])

    def sample(self, batch_size=None, beta=0.5):
        """
        sample a mini batch from experience replay
        :param global_step: now training step
        :return: experience, list, samples
        :return: w, list, weights
        :return: rank_e_id, list, samples id, used for update priority
        """
        if batch_size is None:
            batch_size = self.batch_size

        dist_index = max(math.floor(self.record_size / self.size * self.partition_num), 1)
            # issue 1 by @camigord
        partition_size = math.floor(self.size / self.partition_num)
        partition_max = dist_index * partition_size
        distribution = self.distributions[dist_index]
        rank_list = []
        # sample from k segments
        for n in range(1, batch_size + 1):
            if distribution['strata_ends'][n] + 1 >= distribution['strata_ends'][n + 1]:
                index = distribution['strata_ends'][n] + 1
            else:
                in_storage = False
                while not in_storage:
                    index = random.randint(distribution['strata_ends'][n] + 1,
                                           distribution['strata_ends'][n + 1])
                    try:
                        in_storage = self.priority_queue.priority_to_experience([index])[0] <= self.record_size
                    except:
                        in_storage = False
            rank_list.append(index)


        # find all alpha pow, notice that pdf is a list, start from 0
        alpha_pow = [distribution['pdf'][v - 1] for v in rank_list]
        # w = (N * P(i)) ^ (-beta) / max w
        w = np.power(np.array(alpha_pow) * partition_max, -beta)
        w_max = max(w)
        w = np.divide(w, w_max)
        # rank list is priority id
        # convert to experience id
        rank_e_id = self.priority_queue.priority_to_experience(rank_list)
        # get experience id according rank_e_id
        experience = self.retrieve(rank_e_id)
        return tuple(list(experience) + [w, rank_e_id])
