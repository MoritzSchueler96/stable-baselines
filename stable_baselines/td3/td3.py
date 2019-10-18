import sys
import time
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.her import HindsightExperienceReplayWrapper
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.math_util import safe_mean, unscale_action, scale_action
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.buffers import ReplayBuffer, DiscrepancyReplayBuffer, StableReplayBuffer, PrioritizedReplayBuffer, DRRecurrentReplayBuffer
from stable_baselines.td3.policies import TD3Policy, RecurrentPolicy
from stable_baselines import logger
from stable_baselines.common.schedules import ExponentialSchedule

test = False
class TD3(OffPolicyRLModel):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/pdf/1802.09477.pdf
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: (TD3Policy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values and Actor networks)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update" of the target networks, between 0 and 1)
    :param policy_delay: (int) Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param action_noise: (ActionNoise) the action noise type. Cf DDPG for the different action noise type.
    :param target_policy_noise: (float) Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: (float) Limit for absolute value of target policy smoothing noise.
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param gradient_steps: (int) How many gradient update after each step
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for TD3 normally but can help exploring when using HER + TD3.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on TD3 logging for now
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 buffer_type=ReplayBuffer, prioritization_starts=0, beta_schedule=None,
                 learning_starts=100, train_freq=100, gradient_steps=100, batch_size=128,
                 tau=0.005, policy_delay=2, action_noise=None, action_l2_scale=0,
                 target_policy_noise=0.2, target_noise_clip=0.5,
                 random_exploration=0.0, verbose=0, write_freq=1, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None, time_aware=False):
        super(TD3, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose, write_freq=write_freq,
                                  policy_base=TD3Policy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.prioritization_starts = prioritization_starts
        self.beta_schedule = beta_schedule
        self.buffer_is_prioritized = buffer_type.__name__ in ["PrioritizedReplayBuffer", "RankPrioritizedReplayBuffer"]
        self.loss_history = None
        self.buffer_type = buffer_type
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.action_l2_scale = action_l2_scale
        self.random_exploration = random_exploration
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.time_aware = time_aware

        self.graph = None
        self.replay_buffer = None
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy_tf = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.is_weights_ph = None
        self.step_ops = None
        self.target_ops = None
        self.infos_names = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.policy_out = None
        self.policy_train_op = None
        self.policy_loss = None

        self.recurrent_policy = getattr(self.policy, "recurrent", False)
        if self.recurrent_policy:
            self.policy_tf_act = None
            self.policy_act = None
            self.buffer_type = DRRecurrentReplayBuffer
            self.goal_ph = None
            self.action_prev_ph = None
            self.my_ph = None
            self.pi_state_ph = None
            self.qf1_state_ph = None
            self.qf2_state_ph = None
            self.pi_state = None
            self.qf1_state = None
            self.qf2_state = None
            self.act_ops = None

        self.active_sampling = False

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        policy_out = unscale_action(self.action_space, self.policy_out)
        return policy.obs_ph, self.actions_ph, policy_out

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

                self.buffer_is_prioritized = self.buffer_type.__name__ in ["PrioritizedReplayBuffer", "RankPrioritizedReplayBuffer"]

                if self.replay_buffer is None:
                    if self.buffer_is_prioritized:
                        if self.num_timesteps is not None and self.prioritization_starts > self.num_timesteps or self.prioritization_starts > 0:
                            self.replay_buffer = ReplayBuffer(self.buffer_size)
                        else:
                            buffer_kw = {"size": self.buffer_size, "alpha": 0.7}
                            if self.buffer_type.__name__ == "RankPrioritizedReplayBuffer":
                                buffer_kw.update({"learning_starts": self.prioritization_starts, "batch_size": self.batch_size})
                            self.replay_buffer = self.buffer_type(**buffer_kw)
                    else:
                        self.replay_buffer = self.buffer_type(self.buffer_size, episode_max_len=300, scan_length=0)

                #self.replay_buffer = DiscrepancyReplayBuffer(self.buffer_size, scorer=self.policy_tf.get_q_discrepancy)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    if self.recurrent_policy:
                        self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                     n_batch=self.batch_size, n_steps=self.batch_size, **self.policy_kwargs)
                        self.policy_tf_act = self.policy(self.sess, self.observation_space, self.action_space, n_batch=1,
                                                           **self.policy_kwargs)
                        self.target_policy_tf = self.policy(self.sess, self.observation_space, self.action_space, n_batch=self.batch_size, n_steps=self.batch_size,
                                                            **self.policy_kwargs)
                    else:
                        self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                     **self.policy_kwargs)
                        self.target_policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                            **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy_tf.obs_ph
                    self.processed_next_obs_ph = self.target_policy_tf.processed_obs
                    self.action_target = self.target_policy_tf.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                    if self.recurrent_policy:
                        self.goal_ph = self.policy_tf.goal_ph
                        self.action_prev_ph = self.policy_tf.action_prev_ph
                        self.pi_state_ph = self.policy_tf.pi_state_ph
                        self.qf1_state_ph = self.policy_tf.qf1_state_ph
                        self.qf2_state_ph = self.policy_tf.qf2_state_ph
                        self.my_ph = self.policy_tf.my_ph
                        self.dones_ph = self.policy_tf.dones_ph
                        self.obs_rnn_ph = self.policy_tf.obs_rnn_ph

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    if self.recurrent_policy:
                        self.policy_out = policy_out = self.policy_tf.make_actor(self.processed_obs_ph, self.goal_ph,
                                                                                 self.obs_rnn_ph, self.action_prev_ph)
                        self.policy_act = policy_act = self.policy_tf_act.make_actor(reuse=True)
                        # Use two Q-functions to improve performance by reducing overestimation bias
                        qf1, qf2 = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph, self.goal_ph,
                                                               self.my_ph, self.obs_rnn_ph, self.action_prev_ph)
                        # Q value when following the current policy
                        qf1_pi, qf2_pi = self.policy_tf.make_critics(self.processed_obs_ph, policy_out, self.goal_ph,
                                                                     self.my_ph, self.obs_rnn_ph, self.action_prev_ph,
                                                                     reuse=True)

                        train_params = [var for var in tf_util.get_trainable_vars("model/pi") if "act" not in var.name]
                        act_params = [var for var in tf_util.get_trainable_vars("model/pi") if "act" in var.name]
                        self.act_ops = [
                            tf.assign(act, train)
                            for act, train in zip(act_params, train_params)
                        ]

                    else:
                        self.policy_out = policy_out = self.policy_tf.make_actor(self.processed_obs_ph)
                        # Use two Q-functions to improve performance by reducing overestimation bias
                        qf1, qf2 = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph)
                        # Q value when following the current policy
                        qf1_pi, qf2_pi = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                policy_out, reuse=True)

                with tf.variable_scope("target", reuse=False):
                    if self.recurrent_policy:
                        # Create target networks
                        target_policy_out = self.target_policy_tf.make_actor(self.processed_next_obs_ph, self.goal_ph,
                                                                             self.obs_rnn_ph, self.action_prev_ph,
                                                                             dones=self.dones_ph)
                        # Target policy smoothing, by adding clipped noise to target actions
                        target_noise = tf.random_normal(tf.shape(target_policy_out), stddev=self.target_policy_noise)
                        target_noise = tf.clip_by_value(target_noise, -self.target_noise_clip, self.target_noise_clip)
                        # Clip the noisy action to remain in the bounds [-1, 1] (output of a tanh)
                        noisy_target_action = tf.clip_by_value(target_policy_out + target_noise, -1, 1)
                        # Q values when following the target policy
                        qf1_target, qf2_target = self.target_policy_tf.make_critics(self.processed_next_obs_ph,
                                                                                    noisy_target_action,
                                                                                    self.goal_ph, self.my_ph,
                                                                                    self.obs_rnn_ph,
                                                                                    self.action_prev_ph,
                                                                                    dones=self.dones_ph)
                    else:
                        # Create target networks
                        target_policy_out = self.target_policy_tf.make_actor(self.processed_next_obs_ph)
                        # Target policy smoothing, by adding clipped noise to target actions
                        target_noise = tf.random_normal(tf.shape(target_policy_out), stddev=self.target_policy_noise)
                        target_noise = tf.clip_by_value(target_noise, -self.target_noise_clip, self.target_noise_clip)
                        # Clip the noisy action to remain in the bounds [-1, 1] (output of a tanh)
                        noisy_target_action = tf.clip_by_value(target_policy_out + target_noise, -1, 1)
                        # Q values when following the target policy
                        qf1_target, qf2_target = self.target_policy_tf.make_critics(self.processed_next_obs_ph,
                                                                                    noisy_target_action)

                # TODO: introduce somwehere here the placeholder for history which updates internal state?
                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two target Q-Values (clipped Double-Q Learning)
                    min_qf_target = tf.minimum(qf1_target, qf2_target)

                    # Targets for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * min_qf_target
                    )

                    # Compute Q-Function loss
                    if self.buffer_is_prioritized:
                        self.is_weights_ph = tf.placeholder(tf.float32, shape=(None, 1), name="is_weights")
                        qf1_loss = tf.reduce_mean(self.is_weights_ph * (q_backup - qf1) ** 2)
                        qf2_loss = tf.reduce_mean(self.is_weights_ph * (q_backup - qf2) ** 2)
                    else:
                        qf1_loss = tf.reduce_mean((q_backup - qf1) ** 2)
                        qf2_loss = tf.reduce_mean((q_backup - qf2) ** 2)

                    q_discrepancy = tf.abs(qf1_pi - qf2_pi)
                    qvalues_losses = qf1_loss + qf2_loss

                    rew_loss = tf.reduce_mean(qf1_pi)

                    action_loss = self.action_l2_scale * tf.nn.l2_loss(self.policy_out)

                    # Policy loss: maximise q value
                    self.policy_loss = policy_loss = -rew_loss + action_loss

                    # Policy train op
                    # will be called only every n training steps,
                    # where n is the policy delay
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    pi_var_list = tf_util.get_trainable_vars('model/pi')
                    if self.recurrent_policy:
                        pi_var_list = [var for var in pi_var_list if "act" not in var.name]
                    policy_train_op = policy_optimizer.minimize(policy_loss, var_list=pi_var_list)
                    self.policy_train_op = policy_train_op
                    # Q Values optimizer
                    qvalues_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    qvalues_params = tf_util.get_trainable_vars('model/values_fn/')

                    # Q Values and policy target params
                    source_params = tf_util.get_trainable_vars("model/")
                    target_params = tf_util.get_trainable_vars("target/")

                    if self.recurrent_policy:
                        source_params = [var for var in tf_util.get_trainable_vars("model/") if "act" not in var.name]

                    # Polyak averaging for target variables
                    self.target_ops = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]

                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    train_values_op = qvalues_optimizer.minimize(qvalues_losses, var_list=qvalues_params)

                    self.infos_names = ['qf1_loss', 'qf2_loss']
                    # All ops to call during one training step
                    self.step_ops = [qf1_loss, qf2_loss,
                                     qf1, qf2, train_values_op, q_discrepancy]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar("rew_loss", rew_loss)
                    tf.summary.scalar("action_loss", action_loss)
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = tf_util.get_trainable_vars("model")
                self.target_params = tf_util.get_trainable_vars("target/")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate, update_policy):
        # Sample a batch from the replay buffer

        if self.buffer_is_prioritized and self.num_timesteps >= self.prioritization_starts:
            batch = self.replay_buffer.sample(self.batch_size, beta=self.beta_schedule(self.num_timesteps))
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_weights, batch_idxs = batch
            if len(batch_weights.shape) == 1:
                batch_weights = np.expand_dims(batch_weights, axis=1)
        elif self.recurrent_policy:
            batch = self.replay_buffer.sample(self.batch_size)
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, batch_goals, batch_hist_o, batch_hist_a, batch_mys \
                = batch
        else:
            batch = self.replay_buffer.sample(self.batch_size)
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch
            if self.buffer_is_prioritized:
                batch_weights = np.ones(shape=(self.batch_size, 1))

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate
        }

        if self.recurrent_policy:
            # TODO: does this lose important gradient contributions?
            self.pi_states = None
            rnn_state_reset = np.ones_like(batch_dones)

            feed_dict.update({
                self.my_ph: batch_mys,
                self.goal_ph: batch_goals,
                self.obs_rnn_ph: batch_hist_o,
                self.action_prev_ph: batch_hist_a,
                self.dones_ph: rnn_state_reset,
                self.policy_tf.pi_state_ph: self.policy_tf.initial_state,
                self.policy_tf.qf1_state_ph: self.policy_tf.initial_state,
                self.policy_tf.qf2_state_ph: self.policy_tf.initial_state,
                self.target_policy_tf.pi_state_ph: self.policy_tf.initial_state,
                self.target_policy_tf.qf1_state_ph: self.policy_tf.initial_state,
                self.target_policy_tf.qf2_state_ph: self.policy_tf.initial_state,
                #self.policy_tf.pi_state_ph: np.array([self.policy_tf.initial_state for i in range(self.batch_size)]),
                #self.policy_tf.qf1_state_ph: np.array([self.policy_tf.initial_state for i in range(self.batch_size)]),
                #self.policy_tf.qf2_state_ph: np.array([self.policy_tf.initial_state for i in range(self.batch_size)])
            })

        if self.buffer_is_prioritized:
            feed_dict[self.is_weights_ph] = batch_weights

        step_ops = self.step_ops
        if update_policy:
            # Update policy and target networks
            step_ops = step_ops + [self.policy_train_op, self.target_ops, self.policy_loss]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(step_ops, feed_dict)

        # Unpack to monitor losses
        q_discrepancies = out.pop(5)
        qf1_loss, qf2_loss, *_values = out

        if self.buffer_is_prioritized and self.num_timesteps >= self.prioritization_starts:
            if isinstance(self.replay_buffer, HindsightExperienceReplayWrapper):
                self.replay_buffer.replay_buffer.update_priorities(batch_idxs, q_discrepancies)
            else:
                self.replay_buffer.update_priorities(batch_idxs, q_discrepancies)

        return qf1_loss, qf2_loss, q_discrepancies

    def learn(self, total_timesteps, callback=None,
              log_interval=4, tb_log_name="TD3", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)
        last_replay_update = 0

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            obs = self.env.reset()
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()
            n_updates = 0
            infos_values = []
            self.active_sampling = False
            initial_step = self.num_timesteps

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()

            if self.buffer_is_prioritized and \
                    ((replay_wrapper is not None and self.replay_buffer.replay_buffer.__name__ == "ReplayBuffer")
                     or (replay_wrapper is None and self.replay_buffer.__name__ == "ReplayBuffer")) \
                    and self.num_timesteps >= self.prioritization_starts:
                self._set_prioritized_buffer()

            if self.recurrent_policy:
                done = False
                self.pi_state = self.policy_tf.initial_state
                obs_dict = self.env.convert_obs_to_dict(obs)
                d_goal = obs_dict["desired_goal"]
                obs = np.concatenate([obs_dict["observation"], obs_dict["achieved_goal"]])
                action_prev = np.zeros(shape=self.action_space.shape)

            for step in range(initial_step, total_timesteps):
                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if self.num_timesteps < self.learning_starts or np.random.rand() < self.random_exploration:
                    # actions sampled from action space are from range specific to the environment
                    # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                    unscaled_action = self.env.action_space.sample()
                    action = scale_action(self.action_space, unscaled_action)
                else:
                    if self.recurrent_policy:
                        action, self.pi_state = self.policy_tf_act.step(obs[None], state=self.pi_state, goal=d_goal[None], action_prev=action_prev[None], mask=np.array(done)[None])
                        action = action.flatten()
                    else:
                        action = self.policy_tf.step(obs[None]).flatten()
                    # Add noise to the action, as the policy
                    # is deterministic, this is required for exploration
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # Rescale from [-1, 1] to the correct bounds
                    unscaled_action = unscale_action(self.action_space, action)

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(unscaled_action)

                self.num_timesteps += 1

                # Only stop training if return value is False, not when it is None. This is for backwards
                # compatibility with callbacks that have no return statement.
                if callback.on_step() is False:
                    break

                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                else:
                    # Avoid changing the original ones
                    obs_, new_obs_, reward_ = obs, new_obs, reward

                # Store transition in the replay buffer.
                if self.recurrent_policy:
                    action_prev = action
                    new_obs = self.env.convert_obs_to_dict(new_obs)
                    d_goal = new_obs["desired_goal"]
                    new_obs = np.concatenate([new_obs["observation"], new_obs["achieved_goal"]])
                    if done:
                        self.replay_buffer.add(obs, action, reward, new_obs, done, d_goal, self.env.env.get_simulator_parameters())
                    else:
                        self.replay_buffer.add(obs, action, reward, new_obs, done, d_goal)
                    obs = new_obs
                else:
                    self.replay_buffer.add(obs, action, reward, new_obs, float(done if not self.time_aware else done and info["termination"] != "steps"))
                    obs = new_obs

                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                if ((replay_wrapper is not None and self.replay_buffer.replay_buffer.__name__ == "RankPrioritizedReplayBuffer")\
                        or self.replay_buffer.__name__ == "RankPrioritizedReplayBuffer") and \
                        self.num_timesteps % self.buffer_size == 0:
                    self.replay_buffer.rebalance()

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    self.ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    tf_util.total_episode_reward_logger(self.episode_reward, ep_reward,
                                                        ep_done, writer, self.num_timesteps)

                if self.num_timesteps % self.train_freq == 0:
                    callback.on_rollout_end()

                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                                or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - self.num_timesteps / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        # Note: the policy is updated less frequently than the Q functions
                        # this is controlled by the `policy_delay` parameter
                        step_writer = writer if grad_step % self.write_freq == 0 else None
                        mb_infos_vals.append(
                            self._train_step(step, step_writer, current_lr, (step + grad_step) % self.policy_delay == 0))

                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                    callback.on_rollout_start()


                episode_rewards[-1] += reward_
                if done:
                    if isinstance(self.replay_buffer, DiscrepancyReplayBuffer) and n_updates - last_replay_update >= 5000:
                        self.replay_buffer.update_priorities()
                        last_replay_update = n_updates
                    if self.action_noise is not None:
                        self.action_noise.reset()
                    if not isinstance(self.env, VecEnv):
                        if self.active_sampling:
                            sample_obs, sample_state = self.env.get_random_initial_states(25)
                            obs_discrepancies = self.policy_tf.get_q_discrepancy(sample_obs)
                            obs = self.env.reset(**sample_state[np.argmax(obs_discrepancies)])
                        else:
                            obs = self.env.reset()
                            obs_dict = self.env.convert_obs_to_dict(obs)
                            d_goal = obs_dict["desired_goal"]
                            obs = np.concatenate([obs_dict["observation"], obs_dict["achieved_goal"]])
                            action_prev = np.zeros(shape=self.action_space.shape)
                    episode_rewards.append(0.0)

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)

                self.num_timesteps += 1

                if self.buffer_is_prioritized and \
                        ((replay_wrapper is not None and self.replay_buffer.replay_buffer.__name__ == "ReplayBuffer")
                         or (replay_wrapper is None and self.replay_buffer.__name__ == "ReplayBuffer"))\
                        and self.num_timesteps >= self.prioritization_starts:
                    self._set_prioritized_buffer()

                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []

            callback.on_training_end()
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        _ = np.array(observation)

        if actions is not None:
            raise ValueError("Error: TD3 does not have action probabilities.")

        # here there are no action probabilities, as DDPG does not use a probability distribution
        warnings.warn("Warning: action probability is meaningless for TD3. Returning None")
        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        state = None
        if self.recurrent_policy:
            actions, state = self.policy_tf_act.step(observation, state, mask)
        else:
            actions = self.policy_tf.step(observation)

        if self.action_noise is not None and not deterministic:
            actions = np.clip(actions + self.action_noise(), -1, 1)

        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.action_space, actions)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, state

    def _get_env(self):
        env = self.env
        env = env.env

        return env

    def _set_prioritized_buffer(self):
        buffer_kw = {"size": self.buffer_size, "alpha": 0.7}
        if self.buffer_type.__name__ == "RankPrioritizedReplayBuffer":
            buffer_kw.update({"learning_starts": self.prioritization_starts, "batch_size": self.batch_size})
        r_buf = self.buffer_type(**buffer_kw)

        for i, transition in enumerate(self.replay_buffer._storage):
            r_buf.add(*transition)
            r_buf.update_priorities([i], self.policy_tf.get_q_discrepancy(transition[0])[0])
        if r_buf.__name__ == "RankPrioritizedReplayBuffer":
            r_buf.rebalance()
        if isinstance(self.replay_buffer, HindsightExperienceReplayWrapper):
            self.replay_buffer.replay_buffer = r_buf
        else:
            self.replay_buffer = r_buf
        self.learning_rate = get_schedule_fn(self.learning_rate(1) / 4)  # TODO: will not work with non-constant
        self.beta_schedule = get_schedule_fn(self.beta_schedule)
        print("Enabled prioritized replay buffer")

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False):
        data = {
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            # Should we also store the replay buffer?
            # this may lead to high memory usage
            # with all transition inside
            "replay_buffer": self.replay_buffer,
            "policy_delay": self.policy_delay,
            "target_noise_clip": self.target_noise_clip,
            "target_policy_noise": self.target_policy_noise,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "num_timesteps": self.num_timesteps
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
