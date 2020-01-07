import sys
import time
import multiprocessing
from collections import deque
import warnings

import numpy as np
import tensorflow as tf

from stable_baselines.her import HindsightExperienceReplayWrapper, HERGoalEnvWrapper
from stable_baselines.a2c.utils import total_episode_reward_logger
from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.deepq.replay_buffer import ReplayBuffer, DiscrepancyReplayBuffer, StableReplayBuffer, PrioritizedReplayBuffer, DRRecurrentReplayBuffer
from stable_baselines.ppo2.ppo2 import safe_mean, get_schedule_fn
from stable_baselines.sac.sac import get_vars
from stable_baselines.td3.policies import TD3Policy, RecurrentPolicy, DRPolicy
from stable_baselines import logger
from stable_baselines.common.schedules import ExponentialSchedule


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
    :param target_policy_noise: (float) Standard deviation of gaussian noise added to target policy
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
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=3e-4, buffer_size=50000,
                 buffer_type=ReplayBuffer, buffer_kwargs=None, prioritization_starts=0, beta_schedule=None,
                 learning_starts=100, train_freq=100, gradient_steps=100, batch_size=128,
                 tau=0.005, policy_delay=2, action_noise=None, action_l2_scale=0,
                 target_policy_noise=0.2, target_noise_clip=0.5,
                 random_exploration=0.0, verbose=0, write_freq=1, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, time_aware=False,
                 expert=None, expert_scale=0, expert_q_filter=None,
                 expert_filtering_starts=0, pretrain_expert=False, expert_value_path=None, clip_q_target=None,
                 q_filter_scale_noise=False, reward_transformation=None,
                 initialize_value_path=None, exploration="agent", n_step_returns=1):
        super(TD3, self).__init__(policy=policy, env=env, replay_buffer=None, verbose=verbose, write_freq=write_freq,
                                  policy_base=TD3Policy, requires_vec_env=False, policy_kwargs=policy_kwargs)

        self.prioritization_starts = prioritization_starts
        self.beta_schedule = beta_schedule
        self.buffer_is_prioritized = buffer_type.__name__ in ["PrioritizedReplayBuffer", "RankPrioritizedReplayBuffer"]
        self.loss_history = None
        self.buffer_type = buffer_type
        self.buffer_size = buffer_size
        self.buffer_kwargs = buffer_kwargs
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
        self.expert = expert
        self.expert_scale = expert_scale
        self.pretrain_expert = pretrain_expert
        self.expert_filtering_starts = expert_filtering_starts
        self.expert_value_path = expert_value_path
        self.expert_scale_ph = None
        assert expert_q_filter in [None, "expert", "own"]
        self.expert_q_filter = expert_q_filter
        self.initialize_value_path = initialize_value_path
        self.exploration = exploration

        self.clip_q_target = clip_q_target
        assert clip_q_target is None or len(clip_q_target) == 2
        
        self.reward_transformation = reward_transformation

        self.n_step_returns = n_step_returns

        self.graph = None
        self.replay_buffer = None
        self.episode_reward = None
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
        self.gamma_ph = None

        self.recurrent_policy = getattr(self.policy, "recurrent", False)
        if self.recurrent_policy:
            self.policy_tf_act = None
            self.policy_act = None
            self.pi_state_ph = None
            self.qf1_state_ph = None
            self.qf2_state_ph = None
            self.pi_state = None
            self.qf1_state = None
            self.qf2_state = None
            self.act_ops = None
            self.dones_ph = None

        self.train_extra_phs = {}

        self.expert_actions_ph = None
        self.q_filter_ph = None
        self.q_filter_moving_average = None
        self.q_filter_activation_ph = None
        self.q_filter_scale_noise = q_filter_scale_noise

        self.active_sampling = False

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        # Rescale
        policy_out = self.policy_out * np.abs(self.action_space.low)
        return policy.obs_ph, self.actions_ph, policy_out

    def setup_model(self):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                n_cpu = multiprocessing.cpu_count()
                if sys.platform == 'darwin':
                    n_cpu //= 2
                self.sess = tf_util.make_session(num_cpu=n_cpu, graph=self.graph)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    if self.recurrent_policy:
                        import inspect
                        policy_tf_args = inspect.signature(self.policy).parameters

                        policy_tf_kwargs = {}
                        if "my_size" in policy_tf_args:
                            policy_tf_kwargs["my_size"] = len(self._get_env_parameters())
                        if "goal_size" in policy_tf_args:
                            policy_tf_kwargs["goal_size"] = self.env.goal_dim

                        scan_length = self.buffer_kwargs.get("scan_length", 0)

                        self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                     n_batch=self.batch_size, n_steps=scan_length + 1,
                                                     **policy_tf_kwargs, **self.policy_kwargs)
                        self.policy_tf_act = self.policy(self.sess, self.observation_space, self.action_space,
                                                         n_batch=1, **policy_tf_kwargs,
                                                         **self.policy_kwargs)
                        self.target_policy_tf = self.policy(self.sess, self.observation_space, self.action_space,
                                                            n_batch=self.batch_size,
                                                            n_steps=scan_length + 1, **policy_tf_kwargs,
                                                            **self.policy_kwargs)

                        for ph_name in self.policy_tf.extra_phs:
                            if "target_" in ph_name:
                                self.train_extra_phs[ph_name] = getattr(self.target_policy_tf, ph_name.replace("target_", "") + "_ph")
                            else:
                                self.train_extra_phs[ph_name] = getattr(self.policy_tf, ph_name + "_ph")

                        self.pi_state_ph = self.policy_tf.pi_state_ph
                        self.qf1_state_ph = self.policy_tf.qf1_state_ph
                        self.qf2_state_ph = self.policy_tf.qf2_state_ph
                        self.dones_ph = self.policy_tf.dones_ph
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

                    if self.expert is not None:
                        if self.q_filter_scale_noise:
                            self.q_filter_activation_ph = tf.placeholder(tf.float32, [], name="q_filter_activation_ph")
                            self.q_filter_moving_average = deque(maxlen=1000)
                        self.expert_actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                            name="expert_actions")

                self.buffer_is_prioritized = self.buffer_type.__name__ in ["PrioritizedReplayBuffer",
                                                                           "RankPrioritizedReplayBuffer"]

                if self.replay_buffer is None:
                    if self.buffer_is_prioritized:
                        if self.num_timesteps is not None and self.prioritization_starts > self.num_timesteps or self.prioritization_starts > 0:
                            self.replay_buffer = ReplayBuffer(self.buffer_size)
                        else:
                            buffer_kw = {"size": self.buffer_size, "alpha": 0.7}
                            if self.buffer_type.__name__ == "RankPrioritizedReplayBuffer":
                                buffer_kw.update(
                                    {"learning_starts": self.prioritization_starts, "batch_size": self.batch_size})
                            self.replay_buffer = self.buffer_type(**buffer_kw)
                    else:
                        replay_buffer_kw = {"size": self.buffer_size, **self.buffer_kwargs}
                        if self.recurrent_policy:
                            replay_buffer_kw["extra_data_names"] = self.policy_tf.extra_data_names
                        self.replay_buffer = self.buffer_type(**replay_buffer_kw)

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    if self.recurrent_policy:
                        if self.pretrain_expert:
                            self.policy_out = policy_out = self.expert_actions_ph
                        else:
                            actor_args = inspect.signature(self.policy_tf.make_actor).parameters
                            actor_kws = {k: v for k, v in self.train_extra_phs.items() if k in actor_args}
                            self.policy_out = policy_out = self.policy_tf.make_actor(self.processed_obs_ph, **actor_kws)
                            self.policy_act = policy_act = self.policy_tf_act.make_actor(reuse=True)
                        # Use two Q-functions to improve performance by reducing overestimation bias
                        critic_args = inspect.signature(self.policy_tf.make_critics).parameters
                        critic_kws = {k: v for k, v in self.train_extra_phs.items() if k in critic_args}
                        qf1, qf2 = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph, **critic_kws)
                        # Q value when following the current policy
                        qf1_pi, qf2_pi = self.policy_tf.make_critics(self.processed_obs_ph, policy_out, **critic_kws,
                                                                     reuse=True)

                        if self.expert is not None:
                            qf1_expert, qf2_expert = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                                 self.expert_actions_ph, **critic_kws,
                                                                                 reuse=True)
                    else:
                        if self.pretrain_expert:
                            self.policy_out = policy_out = self.expert_actions_ph
                        else:
                            self.policy_out = policy_out = self.policy_tf.make_actor(self.processed_obs_ph)
                        # Use two Q-functions to improve performance by reducing overestimation bias
                        qf1, qf2 = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph)
                        # Q value when following the current policy
                        qf1_pi, qf2_pi = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                policy_out, reuse=True) 
                
                if self.expert_value_path is not None:
                    _, expert_params = self._load_from_file(self.expert_value_path)
                    expert_params = [val for key, val in expert_params.items() if "model/values_fn" in key]
                    if self.expert is not None:
                        if self.expert_q_filter == "own":
                            with tf.variable_scope("model", reuse=True):
                                qf1_expert, qf2_expert = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                                     self.expert_actions_ph, reuse=True)
                        elif self.expert_q_filter == "expert":
                            with tf.variable_scope("expert", reuse=False):
                                self.expert_tf = self.policy(self.sess, self.observation_space, self.action_space, **self.policy_kwargs)
                                qf1_expert, qf2_expert = self.expert_tf.make_critics(self.processed_obs_ph, self.expert_actions_ph)
                        
                            expert_init_op = [
                                tf.assign(target, source)
                                for target, source in zip(get_vars("expert"), expert_params)
                            ]
                elif self.expert is not None:
                    with tf.variable_scope("model", reuse=True):
                        qf1_expert, qf2_expert = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                     self.expert_actions_ph, reuse=True)

                with tf.variable_scope("target", reuse=False):
                    # Create target networks
                    if self.pretrain_expert:
                        target_policy_out = policy_out
                    else:
                        if self.recurrent_policy:
                            target_policy_out = self.target_policy_tf.make_actor(self.processed_next_obs_ph,
                                                                                 **actor_kws,
                                                                                 dones=self.dones_ph)
                        else:
                            target_policy_out = self.target_policy_tf.make_actor(self.processed_next_obs_ph)
                    # Target policy smoothing, by adding clipped noise to target actions
                    if self.target_policy_noise > 0:
                        if self.q_filter_scale_noise:
                            target_noise = tf.random_normal(tf.shape(target_policy_out), stddev=self.target_policy_noise * self.q_filter_activation_ph)
                        else:
                            target_noise = tf.random_normal(tf.shape(target_policy_out), stddev=self.target_policy_noise)
                        target_noise = tf.clip_by_value(target_noise, -self.target_noise_clip, self.target_noise_clip)
                        # Clip the noisy action to remain in the bounds [-1, 1] (output of a tanh)
                        noisy_target_action = tf.clip_by_value(target_policy_out + target_noise, -1, 1)
                    else:
                        noisy_target_action = target_policy_out
                    # Q values when following the target policy
                    if self.recurrent_policy:
                        qf1_target, qf2_target = self.target_policy_tf.make_critics(self.processed_next_obs_ph,
                                                                                    noisy_target_action,
                                                                                    dones=self.dones_ph,
                                                                                    **critic_kws)
                    else:
                        qf1_target, qf2_target = self.target_policy_tf.make_critics(self.processed_next_obs_ph,
                                                                                noisy_target_action)

                if self.pretrain_expert is not None:
                    policy_pre_activation = policy_out
                else:
                    policy_pre_activation = self.policy_tf.policy_pre_activation

                # TODO: introduce somwehere here the placeholder for history which updates internal state?
                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two target Q-Values (clipped Double-Q Learning)
                    min_qf_target = tf.minimum(qf1_target, qf2_target)

                    if self.n_step_returns == 1:
                        # Targets for Q value regression
                        q_backup = tf.stop_gradient(
                            self.rewards_ph + (1 - self.terminals_ph) * self.gamma * min_qf_target
                        )
                    else:
                        self.gamma_ph = tf.placeholder(tf.float32, shape=(None, 1), name='gamma_ph')
                        q_backup = tf.stop_gradient(
                            self.rewards_ph + (1 - self.terminals_ph) * tf.pow(self.gamma, self.gamma_ph) * min_qf_target
                        )

                    if self.clip_q_target is not None:
                        q_backup = tf.clip_by_value(q_backup, self.clip_q_target[0], self.clip_q_target[1], name="q_backup_clipped")

                    # Compute Q-Function loss
                    if self.buffer_is_prioritized:
                        self.train_extra_phs["is_weights"] = tf.placeholder(tf.float32, shape=(None, 1), name="is_weights")
                        qf1_loss = tf.reduce_mean(self.is_weights_ph * (q_backup - qf1) ** 2)
                        qf2_loss = tf.reduce_mean(self.is_weights_ph * (q_backup - qf2) ** 2)
                    else:
                        qf1_loss = tf.reduce_mean((q_backup - qf1) ** 2)
                        qf2_loss = tf.reduce_mean((q_backup - qf2) ** 2)

                    qvalues_losses = qf1_loss + qf2_loss

                    rew_loss = tf.reduce_mean(qf1_pi)

                    action_loss = self.action_l2_scale * tf.nn.l2_loss(policy_pre_activation)

                    # Policy loss: maximise q value
                    self.policy_loss = policy_loss = -rew_loss + action_loss
                    if self.expert is not None and not self.pretrain_expert:
                        action_difference_norm = tf.norm(self.expert_actions_ph - policy_out, ord="euclidean", axis=1,
                                                         keepdims=True)
                        safe_x = tf.where(action_difference_norm > 1e-5,
                                          action_difference_norm,
                                          tf.zeros([self.batch_size, 1]))
                        self.expert_scale_ph = tf.placeholder(tf.float32, [], name="expert_scale_ph")
                        if self.expert_q_filter is not None:
                            self.q_filtering_disabled_ph = tf.placeholder(tf.bool, [], name="q_filter_disabled_ph")
                            #qf1_pi = tf.Print(qf1_pi, [qf1_pi], "Qf1_pi: ")
                            #qf1_expert = tf.Print(qf1_expert, [qf1_expert], "Qf1_expert: ")
                            expert_action_best = tf.logical_or(qf1_pi < qf1_expert, qf2_pi < qf2_expert)
                            apply_bc_loss = tf.logical_or(expert_action_best, self.q_filtering_disabled_ph)
                            bc_loss = tf.where(apply_bc_loss,
                                               safe_x,
                                               tf.zeros([self.batch_size, 1]))

                            q_filter_activation = tf.reduce_mean(tf.cast(expert_action_best, dtype=tf.float32))
                        else:
                            bc_loss = tf.reduce_mean(safe_x)
                        bc_loss = self.expert_scale_ph * tf.reduce_mean(bc_loss) # TODO: look into issue with NaN in tf.where gradient (https://stackoverflow.com/questions/33712178/tensorflow-nan-bug/42497444#42497444)
                        #bc_loss = tf.Print(bc_loss, [tf.gradients(bc_loss, [self.expert_actions_ph])[0]])
                        self.policy_loss = policy_loss = -rew_loss + action_loss + bc_loss

                    # Policy train op
                    # will be called only every n training steps,
                    # where n is the policy delay

                    if not self.pretrain_expert:
                        policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                        policy_vars = get_vars("model/pi") + get_vars("model/shared")
                        policy_train_op = policy_optimizer.minimize(policy_loss, var_list=policy_vars)
                        self.policy_train_op = policy_train_op

                    # Q Values optimizer
                    qvalues_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    qvalues_params = get_vars('model/values_fn/') + get_vars("model/shared/")

                    # Q Values and policy target params
                    source_params = get_vars("model/")
                    target_params = get_vars("target/")

                    # Polyak averaging for target variables
                    self.target_ops = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    if self.initialize_value_path is not None:
                        if self.initialize_value_path == "expert":
                            source_init_op = [
                                tf.assign(target, source)
                                for target, source in zip(qvalues_params, expert_params)
                            ]
                        else:
                            _, init_params = self._load_from_file(
                                "/home/eivind/Documents/dev/gym-workshop/gym_models/td3_fetchpap_pretrainMC_noise/checkpoint_model_142001.pkl")
                            init_params = [val for key, val in init_params.items() if "model/values_fn" in key]
                            source_init_op = [
                                tf.assign(target, source)
                                for target, source in zip(qvalues_params, init_params)
                            ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    train_values_op = qvalues_optimizer.minimize(qvalues_losses, var_list=qvalues_params)

                    self.infos_names = ['qf1_loss', 'qf2_loss']
                    # All ops to call during one training step
                    self.step_ops = [qf1_loss, qf2_loss, qf1, qf2, train_values_op]

                    if self.q_filter_scale_noise:
                        self.step_ops.append(q_filter_activation)

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar("rew_loss", rew_loss)
                    tf.summary.scalar("action_loss", action_loss)
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                    if self.expert is not None and not self.pretrain_expert:
                        tf.summary.scalar("bc_loss", bc_loss)
                        if self.expert_q_filter is not None:
                            tf.summary.scalar("Q-filter", q_filter_activation)

                # Retrieve parameters that must be saved
                self.params = get_vars("model")
                self.target_params = get_vars("target/")

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    if self.initialize_value_path is not None:
                        self.sess.run(source_init_op)
                    if self.expert_value_path is not None and self.expert_q_filter == "expert":
                        self.sess.run(expert_init_op)
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate, update_policy, viz=False):
        # Sample a batch from the replay buffer
        sample_kw = {}
        if self.buffer_is_prioritized and self.num_timesteps >= self.prioritization_starts:
            sample_kw["beta"] = self.beta_schedule(self.num_timesteps)

        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones, *batch_extra = self.replay_buffer.sample(self.batch_size, **sample_kw)
        batch_extra = batch_extra[0]

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: ~batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate
        }

        if self.n_step_returns > 1:  # TODO: n-step
            feed_dict[self.gamma_ph] = batch_n_steps.reshape(self.batch_size, -1)

        if self.expert is not None:
            if self.pretrain_expert:
                feed_dict[self.expert_actions_ph] = batch_actions
            else:
                feed_dict[self.expert_actions_ph] = batch_expert_actions
                feed_dict[self.expert_scale_ph] = self.expert_scale(self.num_timesteps)
                if self.expert_q_filter is not None:
                    if self.q_filter_scale_noise:
                        feed_dict[self.q_filter_activation_ph] = 1.0 - np.mean(self.q_filter_moving_average)
                    feed_dict[self.q_filtering_disabled_ph] = self.num_timesteps < self.expert_filtering_starts

        if self.recurrent_policy:
            feed_dict.update({
                self.dones_ph: batch_extra["reset"],
                self.pi_state_ph: self.policy_tf.pi_initial_state,
                self.qf1_state_ph: self.policy_tf.qf1_initial_state,
                self.qf2_state_ph: self.policy_tf.qf2_initial_state,
                self.target_policy_tf.pi_state_ph: self.target_policy_tf.pi_initial_state,
                self.target_policy_tf.qf1_state_ph: self.target_policy_tf.qf1_initial_state,
                self.target_policy_tf.qf2_state_ph: self.target_policy_tf.qf2_initial_state
            })

        feed_dict.update({v: batch_extra[k] for k, v in self.train_extra_phs.items()})

        if viz:
            self.q_val_action_viz(batch_obs, batch_actions, vis_dim="2d")
        step_ops = self.step_ops
        if update_policy:
            # Update policy and target networks
            if not self.pretrain_expert:
                step_ops = step_ops + [self.policy_train_op, self.target_ops, self.policy_loss]
            else:
                step_ops = step_ops + [self.target_ops]

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            out = self.sess.run([self.summary] + step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            out = self.sess.run(step_ops, feed_dict)

        # Unpack to monitor losses
        qf1_loss, qf2_loss, *_values = out

        return qf1_loss, qf2_loss

    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=4, tb_log_name="TD3", reset_num_timesteps=True, replay_wrapper=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        last_replay_update = 0

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        if isinstance(self.train_freq, tuple):  # TODO: bug with optuna please FIX
            self.train_freq = self.train_freq[0]
            self.gradient_steps = self.gradient_steps[0]

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn(seed)

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            self.expert_scale = get_schedule_fn(self.expert_scale)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            obs = self.env.reset()

            if getattr(self.env, "norm", False):
                original_obs = self.env.get_original_obs()

            self.episode_reward = np.zeros((1,))
            ep_info_buf = deque(maxlen=100)
            n_updates = 0
            infos_values = []
            self.active_sampling = False
            initial_step = self.num_timesteps
            episode_data = []

            if self.buffer_is_prioritized and \
                    ((replay_wrapper is not None and self.replay_buffer.replay_buffer.__name__ == "ReplayBuffer")
                     or (replay_wrapper is None and self.replay_buffer.__name__ == "ReplayBuffer")) \
                    and self.num_timesteps >= self.prioritization_starts:
                self._set_prioritized_buffer()

            if self.recurrent_policy:
                done = False
                self.pi_state = self.policy_tf_act.initial_state

            if self.q_filter_scale_noise and len(self.q_filter_moving_average) == 0:
                self.q_filter_moving_average.append(1)

            for step in range(initial_step, total_timesteps):
                if callback is not None:
                    # Only stop training if return value is False, not when it is None. This is for backwards
                    # compatibility with callbacks that have no return statement.
                    if callback(locals(), globals()) is False:
                        break

                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if self.expert is not None:
                    if getattr(self.env, "norm", False):
                        original_obs = self.env.get_original_obs()
                    else:
                        original_obs = obs
                    expert_action = self.expert(original_obs[None]).flatten()
                if (self.num_timesteps < self.learning_starts
                        or np.random.rand() < self.random_exploration):
                    # No need to rescale when sampling random action
                    rescaled_action = action = self.env.action_space.sample()
                else:
                    if self.recurrent_policy:
                        action, self.pi_state = self.policy_tf_act.step(obs[None], state=self.pi_state, mask=np.array(done)[None])
                        action = action.flatten()
                    else:
                        if self.pretrain_expert or self.exploration == "expert":
                            action = expert_action
                        elif self.exploration == "q-filter":
                            agent_score, expert_score = self.sess.run([self.qf1_pi, self.qf1_expert], {self.observations_ph: obs[None], self.expert_actions_ph: action[None]})
                            if agent_score >= expert_score:
                                action = self.policy_tf.step(obs[None]).flatten()
                            else:
                                action = expert_action
                        else:
                            action = self.policy_tf.step(obs[None]).flatten()
                    # Add noise to the action, as the policy
                    # is deterministic, this is required for exploration
                    if self.action_noise is not None:
                        action_noise = self.action_noise()
                        if self.q_filter_scale_noise:
                            action_noise *= (1.0 - np.mean(self.q_filter_moving_average))
                        action = np.clip(action + action_noise, -1, 1)
                    # Rescale from [-1, 1] to the correct bounds
                    rescaled_action = action * np.abs(self.action_space.low)

                # TODO: expert action choice here?

                assert action.shape == self.env.action_space.shape

                new_obs, reward, done, info = self.env.step(rescaled_action)

                # TODO: CAN PRECOMPUTE EXPERT Q-VALUE

                if self.reward_transformation is not None:
                    reward = self.reward_transformation(reward)

                # Store transition in the replay buffer.
                extra_data = {}
                if self.time_aware:
                    bootstrap = True
                    if done:
                        info_time_limit = info.get("TimeLimit.truncated", None)
                        bootstrap = info.get("termination", None) == "steps" or \
                                    (info_time_limit is not None and info_time_limit)
                    extra_data["bootstrap"] = bootstrap

                if self.expert is not None and not self.pretrain_expert:
                    if getattr(self.env, "norm", False):  # If norm, first get action from obs (not new_obs)
                        extra_data["expert_action"] = expert_action
                        extra_data["original_obs"] = original_obs
                        original_obs = self.env.get_original_obs()  # get unnormalized new_obs
                        extra_data["original_obs_new"] = original_obs
                    else:
                        extra_data["expert_action"] = expert_action

                if hasattr(self.policy, "collect_data"):
                    extra_data.update(self.policy_tf_act.collect_data(locals(), globals()))
                self.replay_buffer.add(obs, action, reward, new_obs, done, **extra_data)
                episode_data.append({"obs": obs, "action": action, "reward": reward, "new_obs": new_obs, "done": done, **extra_data})
                obs = new_obs

                if ((replay_wrapper is not None and self.replay_buffer.replay_buffer.__name__ == "RankPrioritizedReplayBuffer")\
                        or self.replay_buffer.__name__ == "RankPrioritizedReplayBuffer") and \
                        self.num_timesteps % self.buffer_size == 0:
                    self.replay_buffer.rebalance()

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_reward,
                                                                      ep_done, writer, self.num_timesteps)

                if step % self.train_freq == 0:
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
                            self._train_step(step, step_writer, current_lr, (step + grad_step) % self.policy_delay == 0
                        ,grad_step == 0))

                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                episode_rewards[-1] += reward
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

                    episode_data = []
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
                    if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in ep_info_buf]))
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
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        _ = np.array(observation)

        if actions is not None:
            raise ValueError("Error: TD3 does not have action probabilities.")

        # here there are no action probabilities, as DDPG does not use a probability distribution
        warnings.warn("Warning: action probability is meaningless for TD3. Returning None")
        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        # TODO: Maybe use target_network for test set (HER paper does this, because it is more stable)
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        state = None
        if self.recurrent_policy:
            actions, state = self.policy_tf_act.step(observation, state=state, mask=mask)
        else:
            actions = self.policy_tf.step(observation)

        if self.action_noise is not None and not deterministic:
            actions = np.clip(actions + self.action_noise(), -1, 1)

        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = actions * np.abs(self.action_space.low)  # scale the output for the prediction

        if not vectorized_env:
            actions = actions[0]

        return actions, state

    def _get_env(self):
        env = self.env
        env = env.env

        return env

    def _get_env_parameters(self):
        return np.zeros((37,))
        if isinstance(self.env, HERGoalEnvWrapper):
            return self.env.env.get_simulator_parameters()
        else:
            return self.env.get_simulator_parameters()
        
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

    def save(self, save_path):
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
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs,
            "num_timesteps": self.num_timesteps,
            "buffer_type": self.buffer_type,
            "buffer_kwargs": self.buffer_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save)

    def q_val_action_viz(self, obses, actions, q_func=None, vis_dim="1d"):
        import matplotlib.pyplot as plt
        import math

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array-value)).argmin()
            return array[idx]

        def find_nearest2d(array, value):
            array = np.asarray(array)
            idx = (np.linalg.norm(array-value, axis=1)).argmin()
            return idx

        if q_func is None:
            q_func = self.qf1
        action_ph = self.expert_actions_ph if q_func == self.qf1_expert else self.actions_ph
        if vis_dim == "1d":
            x = np.linspace(-1, 1, 20)
            if len(obses.shape) == 1:
                obses = obses.reshape(1, -1)
            if len(actions.shape) == 1:
                actions = actions.reshape(1, -1)

            data = [{k: [] for k in x} for i in range(actions.shape[1])]

            for obs_i in range(obses.shape[0]):
                for action_i in range(actions.shape[1]):
                    for x_i in x:
                        action_mod = np.zeros(shape=(actions.shape[1],))
                        action_mod[action_i] = x_i
                        action_new = actions[obs_i, action_i] + action_mod
                        if np.abs(actions[obs_i, action_i] + x_i) > 1:
                            continue
                        q_val = self.sess.run(q_func, {self.observations_ph: obses[obs_i, :].reshape(1, -1),
                                                       action_ph: action_new.reshape(1, -1)})[0][0]
                        data[action_i][find_nearest(x, actions[obs_i, action_i] + x_i)].append(q_val)

            num_rows = int(actions.shape[1] // np.sqrt(actions.shape[1]))
            num_cols = math.ceil(actions.shape[1] / 2) if num_rows > 1 else actions.shape[1]
            fig, axs = plt.subplots(num_rows, num_cols)
            axs = axs.reshape(-1)
            for action_i in range(actions.shape[1]):
                axs[action_i].plot(x, [np.mean(data[action_i][x_i]) for x_i in x])
                axs[action_i].set_xlabel("Action deviation")
                axs[action_i].set_ylabel("Q-value")
                axs[action_i].set_title("Action {}".format(action_i))
        elif vis_dim == "2d":
            from mpl_toolkits.mplot3d import Axes3D
            xs, ys = np.mgrid[-2:2:21j, -2:2:21j]
            points = np.dstack([np.ravel(xs), np.ravel(ys)]).reshape(-1, 2)
            data = [[] for i in range(points.shape[0])]
            for obs_i in range(obses.shape[0]):
                action_x, action_y = actions[obs_i, 0] + xs, actions[obs_i, 1] + ys
                actions_i = np.dstack([np.ravel(action_x), np.ravel(action_y)]).reshape(-1, 2)
                valid_actions_idxs = np.where((np.abs(actions_i) <= 1).all(axis=1))[0]
                valid_actions = actions_i[valid_actions_idxs]
                q_vals = self.sess.run(q_func, {self.observations_ph: np.repeat(obses[obs_i, :].reshape(1, -1), valid_actions_idxs.shape[0], axis=0), action_ph: valid_actions})
                #for va_i, valid_action in enumerate(d_actions):
                #    data[find_nearest2d(points, valid_action)].append(q_vals[va_i])
                for va_i, idx in enumerate(valid_actions_idxs):
                    data[idx].append(q_vals[va_i])

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            z = np.array([np.mean(data[i]) for i in range(len(data))])
            ax.plot_surface(xs, ys, z.reshape(xs.shape))
            ax.set_xlabel("Action 0 deviation")
            ax.set_ylabel("Action 1 devation")
            ax.set_zlabel("Q-value")
            #plt.scatter([0.], [0.], z[find_nearest2d(points, [0, 0])] + 1, marker="x")
            plt.plot([0, 0], [0, 0], [np.nanmin(z) - 1, np.nanmax(z) + 1], linewidth=3)

        #plt.show()
        plt.savefig("/home/eivind/Documents/dev/gym-workshop/gym_models/td3_reacher_expert_TRAININGSHAPETEST/q_viz_{}.png".format(self.num_timesteps), format="png", bbox_inches="tight")
