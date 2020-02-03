import tensorflow as tf
import numpy as np
from gym.spaces import Box
import copy
from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy, cnn_1d_extractor
from stable_baselines.sac.policies import mlp
from stable_baselines.a2c.utils import lstm, batch_to_seq, seq_to_batch


class TD3Policy(BasePolicy):
    """
    Policy object that implements a TD3-like actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, scale=False,
                 add_action_ph=False):
        super(TD3Policy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale,
                                        add_action_ph=add_action_ph)
        assert isinstance(ac_space, Box), "Error: the action space must be of type gym.spaces.Box"

        self.qf1 = None
        self.qf2 = None
        self.q_discrepancy = None
        self.policy = None

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        """
        Creates an actor object

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name of the actor
        :return: (TensorFlow Tensor) the output tensor
        """
        raise NotImplementedError

    def make_critics(self, obs=None, action=None, reuse=False,
                     scope="qvalues_fn"):
        """
        Creates the two Q-Values approximator

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param action: (TensorFlow Tensor) The action placeholder
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name
        :return: ([tf.Tensor]) Mean, action and log probability
        """
        raise NotImplementedError

    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) actions
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) actions
        """
        return self.step(obs, state, mask)


class FeedForwardPolicy(TD3Policy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn",
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                reuse=reuse,
                                                scale=(feature_extraction == "cnn" and cnn_extractor == nature_cnn))

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.cnn_vf = self.cnn_kwargs.pop("cnn_vf", True)
        self.reuse = reuse
        if layers is None:
            layers = [64, 64]
        self.layers = layers
        self.obs_module_indices = obs_module_indices
        self.policy_pre_activation = None

        assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

        self.activ_fn = act_fun

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        if self.obs_module_indices is not None:
            obs = tf.gather(obs, self.obs_module_indices["pi"], axis=-1)
        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, name="pi_c1", act_fun=self.activ_fn, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)

            pi_h = mlp(pi_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)

            self.policy_pre_activation = tf.layers.dense(pi_h, self.ac_space.shape[0])
            self.policy = policy = tf.tanh(self.policy_pre_activation)

        return policy

    def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn"):
        if obs is None:
            obs = self.processed_obs

        if self.obs_module_indices is not None:
            obs = tf.gather(obs, self.obs_module_indices["vf"], axis=-1)

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn" and self.cnn_vf:
                critics_h = self.cnn_extractor(obs, name="vf_c1", act_fun=self.activ_fn, **self.cnn_kwargs)
            else:
                critics_h = tf.layers.flatten(obs)

            # Concatenate preprocessed state and action
            qf_h = tf.concat([critics_h, action], axis=-1)

            # Double Q values to reduce overestimation
            with tf.variable_scope('qf1', reuse=reuse):
                qf1_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                qf1 = tf.layers.dense(qf1_h, 1, name="qf1")

            with tf.variable_scope('qf2', reuse=reuse):
                qf2_h = mlp(qf_h, self.layers, self.activ_fn, layer_norm=self.layer_norm)
                qf2 = tf.layers.dense(qf2_h, 1, name="qf2")

            self.qf1 = qf1
            self.qf2 = qf2
            # TODO: assumes that all qf1 and qf2 can never have opposite signs
            #self.q_discrepancy = tf.square(self.qf1 - self.qf2) / tf.square(tf.maximum(self.qf1, self.qf2))
            self.q_discrepancy = tf.abs(self.qf1 - self.qf2)

        return self.qf1, self.qf2

    def step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})

    def get_q_discrepancy(self, obs):
        if isinstance(obs, np.ndarray) and len(obs.shape) == 1: # TODO: check for MLP or CNN policy here
            obs = np.expand_dims(obs, axis=0)
        return self.sess.run(self.q_discrepancy, {self.obs_ph: obs})


class RecurrentPolicy(TD3Policy):
    """
        Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

        :param sess: (TensorFlow session) The current TensorFlow session
        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param n_env: (int) The number of environments to run
        :param n_steps: (int) The number of steps to run for each environment
        :param n_batch: (int) The number of batch to run (n_envs * n_steps)
        :param reuse: (bool) If the policy is reusable or not
        :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
        :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
        :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
        :param layer_norm: (bool) enable layer normalisation
        :param act_fun: (tf.func) the activation function to use in the neural network.
        :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
        """
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, layers, n_env=1, n_steps=1, n_batch=None, reuse=False,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_lstm=128, share_lstm=False, save_state=False,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        super(RecurrentPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                reuse=reuse, add_action_ph=True,
                                                scale=(feature_extraction == "cnn" and cnn_extractor == nature_cnn))

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.cnn_vf = self.cnn_kwargs.pop("cnn_vf", True)
        self.reuse = reuse
        self.layers = layers
        self.obs_module_indices = obs_module_indices

        self.activ_fn = act_fun
        self.n_lstm = n_lstm
        self.share_lstm = share_lstm
        self._obs_ph = self.processed_obs

        assert self.n_batch % self.n_steps == 0, "The batch size must be a multiple of sequence length (n_steps)"
        self._lstm_n_batch = self.n_batch // self.n_steps

        self.action_prev = np.zeros((1, *self.ac_space.shape))

        self._initial_state = np.zeros((self._lstm_n_batch, self.n_lstm * 2), dtype=np.float32)
        if self.share_lstm:
            self.state = None
        else:
            self.pi_state = None
            self.qf1_state = None
            self.qf2_state = None

        with tf.variable_scope("input", reuse=False):
            self.dones_ph = tf.placeholder_with_default(np.zeros((self.n_batch,), dtype=np.float32), (self.n_batch,), name="dones_ph")  # (done t-1)
            if self.share_lstm:
                self.state_ph = tf.placeholder_with_default(self.initial_state, (self._lstm_n_batch, self.n_lstm * 2), name="state_ph")
            else:
                self.pi_state_ph = tf.placeholder_with_default(self.initial_state, (self._lstm_n_batch, self.n_lstm * 2), name="pi_state_ph")
                self.qf1_state_ph = tf.placeholder_with_default(self.initial_state, (self._lstm_n_batch, self.n_lstm * 2), name="qf1_state_ph")
                self.qf2_state_ph = tf.placeholder_with_default(self.initial_state, (self._lstm_n_batch, self.n_lstm * 2), name="qf2_state_ph")

            self.action_prev_ph = tf.placeholder(np.float32, (self.n_batch, *self.ac_space.shape), name="action_prev_ph")

        self.extra_phs = ["action_prev"]
        self.rnn_inputs = ["obs", "action_prev"]
        self.extra_data_names = ["action_prev"]

        self.save_state = save_state
        if self.save_state:
            if self.share_lstm:
                self.extra_data_names = sorted(self.extra_data_names + ["state"])
                self.extra_phs = sorted(self.extra_phs + ["state"])
            else:
                self.extra_data_names = sorted(self.extra_data_names + ["pi_state", "qf1_state", "qf2_state"])
                self.extra_phs = sorted(self.extra_phs + ["pi_state", "qf1_state", "qf2_state"])

    def _process_phs(self, **phs):
        for ph_name, ph_val in phs.items():
            if ph_val is None:
                phs[ph_name] = getattr(self, ph_name + "_ph")
            else:
                try:
                    setattr(self, ph_name + "_ph", ph_val)
                except AttributeError:
                    setattr(self, "_" + ph_name + "_ph", ph_val)

        return phs.values()

    def _make_branch(self, branch_name, input_tensor, dones=None, state_ph=None):
        if branch_name == "lstm":
            for i, fc_layer_units in enumerate(self.layers["lstm"]):
                input_tensor = self.activ_fn(tf.layers.dense(input_tensor, fc_layer_units, name="lstm_fc{}".format(i)))

            input_tensor = batch_to_seq(input_tensor, self._lstm_n_batch, self.n_steps)
            masks = batch_to_seq(dones, self._lstm_n_batch, self.n_steps)
            input_tensor, state = lstm(input_tensor, masks, state_ph, "lstm", n_hidden=self.n_lstm,
                                       layer_norm=self.layer_norm)
            input_tensor = seq_to_batch(input_tensor)

            return input_tensor, state
        else:
            for i, fc_layer_units in enumerate(self.layers[branch_name]):
                input_tensor = self.activ_fn(tf.layers.dense(input_tensor, fc_layer_units, name="{}_fc{}".format(branch_name, i)))

            return input_tensor

    def make_actor(self, ff_phs=None, rnn_phs=None, dones=None, reuse=False, scope="pi"):
        lstm_branch = tf.concat([tf.layers.flatten(ph) for ph in rnn_phs], axis=-1)
        if ff_phs is not None:
            ff_branch = tf.concat([tf.layers.flatten(ph) for ph in ff_phs], axis=-1)

        if dones is None:
            dones = self.dones_ph

        if self.share_lstm:
            with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
                lstm_branch, self.state = self._make_branch("lstm", lstm_branch, dones, self.state_ph)

        with tf.variable_scope(scope, reuse=reuse):
            if self.layers["ff"] is not None:
               ff_branch = self._make_branch("ff", ff_branch)

            if not self.share_lstm:
                lstm_branch, self.pi_state = self._make_branch("lstm", lstm_branch, dones, self.pi_state_ph)

            if ff_phs is not None:
                head = tf.concat([ff_branch, lstm_branch], axis=-1)
            else:
                head = lstm_branch

            head = self._make_branch("head", head)

            self.policy_pre_activation = tf.layers.dense(head, self.ac_space.shape[0])
            self.policy = policy = tf.tanh(self.policy_pre_activation)

        return policy

    def make_critics(self, ff_phs=None, rnn_phs=None, dones=None, reuse=False, scope="values_fn"):
        lstm_branch_in = tf.concat([tf.layers.flatten(ph) for ph in rnn_phs], axis=-1)
        if ff_phs is not None:
            ff_branch_in = tf.concat([tf.layers.flatten(ph) for ph in ff_phs], axis=-1)

        if dones is None:
            dones = self.dones_ph

        self.qf1, self.qf2 = None, None
        self.qf1_state, self.qf2_state = None, None

        if self.share_lstm:
            with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
                lstm_branch_s, self.state = self._make_branch("lstm", lstm_branch_in, dones, self.state_ph)

        with tf.variable_scope(scope, reuse=reuse):
            # Double Q values to reduce overestimation
            for qf_i in range(1, 3):
                with tf.variable_scope('qf{}'.format(qf_i), reuse=reuse):
                    lstm_branch = lstm_branch_in
                    if self.layers["ff"] is not None:
                        ff_branch = self._make_branch("ff", ff_branch_in)
                    elif ff_phs is not None:
                        ff_branch = ff_branch_in

                    if not self.share_lstm:
                        lstm_branch, state = self._make_branch("lstm", lstm_branch, dones,
                                                               getattr(self, "qf{}_state_ph".format(qf_i)))
                        setattr(self, "qf{}_state".format(qf_i), state)
                    else:
                        lstm_branch = lstm_branch_s

                    if ff_phs is not None:
                        head = tf.concat([ff_branch, lstm_branch], axis=-1)
                    else:
                        head = lstm_branch

                    head = self._make_branch("head", head)

                    setattr(self, "qf{}".format(qf_i), tf.layers.dense(head, 1, name="qf{}".format(qf_i)))

        return self.qf1, self.qf2

    def step(self, obs, state=None, mask=None):
        raise NotImplementedError

    @property
    def initial_state(self):
        return self._initial_state

    def collect_data(self, _locals, _globals):
        data = {}
        if self.save_state:
            if self.share_lstm:
                data["state"] = _locals["prev_policy_state"][0, :]
            else:
                data["pi_state"] = _locals["prev_policy_state"][0, :]
                if len(_locals["episode_data"]) == 0:
                    qf1_state, qf2_state = self.initial_state, self.initial_state
                else:
                    qf_feed_dict = {
                        self.qf1_state_ph: _locals["episode_data"][-1]["qf1_state"][None],
                        self.qf2_state_ph: _locals["episode_data"][-1]["qf2_state"][None],
                    }
                    qf_feed_dict.update({getattr(self, data_name + "_ph"): _locals["episode_data"][-1][data_name][None]
                                         for data_name in self.rnn_inputs})
                    qf1_state, qf2_state = self.sess.run([self.qf1_state, self.qf2_state], feed_dict=qf_feed_dict)
                data["qf1_state"] = qf1_state[0, :]
                data["qf2_state"] = qf2_state[0, :]

        if len(_locals["episode_data"]) == 0:
            data["action_prev"] = np.zeros(*self.ac_space.shape, dtype=np.float32)
        else:
            data["action_prev"] = _locals["episode_data"][-1]["action"]

        #data["target_action_prev_rnn"] = _locals["action"]

        return data


class DRPolicy(RecurrentPolicy):
    """
        Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

        :param sess: (TensorFlow session) The current TensorFlow session
        :param ob_space: (Gym Space) The observation space of the environment
        :param ac_space: (Gym Space) The action space of the environment
        :param n_env: (int) The number of environments to run
        :param n_steps: (int) The number of steps to run for each environment
        :param n_batch: (int) The number of batch to run (n_envs * n_steps)
        :param reuse: (bool) If the policy is reusable or not
        :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
        :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
        :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
        :param layer_norm: (bool) enable layer normalisation
        :param act_fun: (tf.func) the activation function to use in the neural network.
        :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
        """
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, goal_size, my_size, n_env=1, n_steps=1, n_batch=None, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_lstm=128, share_lstm=False,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        if layers is None:
            layers = {"ff": [128], "lstm": [128], "head": [128, 128]}
        super().__init__(sess, ob_space, ac_space, layers, n_env, n_steps, n_batch,
                                                reuse=reuse, cnn_extractor=cnn_extractor,
                                                feature_extraction=feature_extraction, n_lstm=n_lstm,
                                                share_lstm=share_lstm, layer_norm=layer_norm, act_fun=act_fun,
                                                obs_module_indices=obs_module_indices, **kwargs)

        with tf.variable_scope("input", reuse=False):
            self.my_ph = tf.placeholder(tf.float32, (None, my_size), name="my_ph")  # the dynamics of the environment

        self.goal_size = goal_size
        self.extra_phs = sorted(self.extra_phs + ["my"])
        self.extra_data_names = sorted(self.extra_data_names + ["my"])

    def make_actor(self, obs_ff=None, obs_rnn=None, action_prev=None, dones=None, reuse=False, scope="pi"):
        if obs_ff is None:
            obs_ff = self.processed_obs
        if obs_rnn is None:
            obs_rnn = self.processed_obs
        if action_prev is None:
            action_prev = self.action_prev_ph

        obs_ff, goal = obs_ff[:, :-self.goal_size], obs_ff[:, -self.goal_size:]
        goal = tf.subtract(goal, obs_ff[:, -self.goal_size:], name="goal_relative")
        obs_rnn = obs_rnn[:, :-self.goal_size]

        ff_phs = [obs_ff, goal]
        rnn_phs = [obs_rnn, action_prev]
        return super().make_actor(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def make_critics(self, obs_ff=None, action_ff=None, my=None, obs_rnn=None, action_prev=None, dones=None, reuse=False, scope="values_fn"):
        if obs_ff is None:
            obs_ff = self.processed_obs
        if action_ff is None:
            action_ff = self.action_ph
        if my is None:
            my = self.my_ph
        if obs_rnn is None:
            obs_rnn = self.processed_obs
        if action_prev is None:
            action_prev = self.action_prev_ph

        obs_ff, goal = obs_ff[:, :-self.goal_size], obs_ff[:, -self.goal_size:]
        goal = tf.subtract(goal, obs_ff[:, -self.goal_size:], name="goal_relative")
        obs_rnn = obs_rnn[:, :-self.goal_size]

        ff_phs = [obs_ff, goal, my, action_ff]
        rnn_phs = [obs_rnn, action_prev]
        return super().make_critics(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def step(self, obs, state=None, action_prev=None, mask=None):
        if state is None:
            state = self.initial_state
        if action_prev is None:
            assert obs.shape[0] == 1
            if mask[0]:
                self.action_prev = np.zeros((1, *self.ac_space.shape))
            action_prev = self.action_prev

        lstm_node = self.state if self.share_lstm else self.pi_state
        state_ph = self.state_ph if self.share_lstm else self.pi_state_ph

        action, out_state = self.sess.run([self.policy, lstm_node],
                            {self.obs_ph: obs, state_ph: state, self.dones_ph: mask,
                            self.action_prev_ph: action_prev})
        self.action_prev = action

        return action, out_state

    def collect_data(self, _locals, _globals, **kwargs):
        data = super().collect_data(_locals, _globals)
        if "my" not in _locals or _locals["ep_data"]:
            data["my"] = _locals["self"].env.get_env_parameters()

        return data


class LstmMlpPolicy(RecurrentPolicy):
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False,
                 layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_lstm=128, share_lstm=False,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        if layers is None:
            layers = {"ff": None, "lstm": [64, 64], "head": []}
        else:
            assert layers["ff"] is None
        super().__init__(sess, ob_space, ac_space, layers, n_env, n_steps, n_batch,
                         reuse=reuse, cnn_extractor=cnn_extractor,
                         feature_extraction=feature_extraction, n_lstm=n_lstm,
                         share_lstm=share_lstm, layer_norm=layer_norm, act_fun=act_fun,
                         obs_module_indices=obs_module_indices, **kwargs)

    def make_actor(self, obs=None, action_prev=None, dones=None, reuse=False, scope="pi"):
        obs, action_prev, dones = self._process_phs(obs=obs, action_prev=action_prev, dones=dones)

        ff_phs = None
        rnn_phs = [obs, action_prev]
        return super().make_actor(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def make_critics(self, obs=None, action=None, action_prev=None, dones=None, reuse=False, scope="values_fn"):
        obs, action, action_prev, dones = self._process_phs(obs=obs, action=action, action_prev=action_prev, dones=dones)

        ff_phs = [action]
        rnn_phs = [obs, action_prev]
        return super().make_critics(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def step(self, obs, action_prev=None, state=None, mask=None):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = np.array([False])
        if action_prev is None:
            assert obs.shape[0] == 1
            if mask[0]:
                self.action_prev = np.zeros((1, *self.ac_space.shape))
            action_prev = self.action_prev

        return self.sess.run([self.policy, self.pi_state],
                             {self.obs_ph: obs, self.action_prev_ph: action_prev,
                              self.pi_state_ph: state, self.dones_ph: mask})


class LstmFFMlpPolicy(RecurrentPolicy):
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False,
                 layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_lstm=128, share_lstm=False,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        if layers is None:
            layers = {"ff": [64], "lstm": [64, 64], "head": []}

        super().__init__(sess, ob_space, ac_space, layers, n_env, n_steps, n_batch,
                         reuse=reuse, cnn_extractor=cnn_extractor,
                         feature_extraction=feature_extraction, n_lstm=n_lstm,
                         share_lstm=share_lstm, layer_norm=layer_norm, act_fun=act_fun,
                         obs_module_indices=obs_module_indices, **kwargs)

    def make_actor(self, obs=None, action_prev=None, dones=None, reuse=False, scope="pi"):
        obs, action_prev, dones = self._process_phs(obs=obs, action_prev=action_prev, dones=dones)

        ff_phs = [obs]
        rnn_phs = [obs, action_prev]
        return super().make_actor(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def make_critics(self, obs=None, action=None, action_prev=None, dones=None, reuse=False, scope="values_fn"):
        obs, action, action_prev, dones = self._process_phs(obs=obs, action=action, action_prev=action_prev, dones=dones)

        ff_phs = [obs, action]
        rnn_phs = [obs, action_prev]
        return super().make_critics(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def step(self, obs, action_prev=None, state=None, mask=None):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = np.array([False])
        if action_prev is None:
            assert obs.shape[0] == 1
            if mask[0]:
                self.action_prev = np.zeros((1, *self.ac_space.shape))
            action_prev = self.action_prev

        return self.sess.run([self.policy, self.pi_state],
                             {self.obs_ph: obs, self.action_prev_ph: action_prev,
                              self.pi_state_ph: state, self.dones_ph: mask})


class RecurrentActorCriticPolicy(ActorCriticPolicy):
    """
    Actor critic policy object uses a previous state in the computation for the current step.
    NOTE: this class is not limited to recurrent neural network policies,
    see https://github.com/hill-a/stable-baselines/issues/241

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param state_shape: (tuple<int>) shape of the per-environment state space.
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 state_shape, reuse=False, scale=False):
        super(RecurrentActorCriticPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                         n_batch, reuse=reuse, scale=scale)

        with tf.variable_scope("input", reuse=False):
            self._dones_ph = tf.placeholder(tf.float32, (n_batch,), name="dones_ph")  # (done t-1)
            state_ph_shape = (self.n_env,) + tuple(state_shape)
            self._states_ph = tf.placeholder(tf.float32, state_ph_shape, name="states_ph")

        initial_state_shape = (self.n_env,) + tuple(state_shape)
        self._initial_state = np.zeros(initial_state_shape, dtype=np.float32)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def dones_ph(self):
        """tf.Tensor: placeholder for whether episode has terminated (done), shape (self.n_batch, ).
        Internally used to reset the state before the next episode starts."""
        return self._dones_ph

    @property
    def states_ph(self):
        """tf.Tensor: placeholder for states, shape (self.n_env, ) + state_shape."""
        return self._states_ph

    @abstractmethod
    def value(self, obs, state=None, mask=None):
        """
        Cf base class doc.
        """
        raise NotImplementedError


class LstmPolicy(RecurrentActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTMs.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture. Notation similar to the
        format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None,
                 net_arch=None, act_fun=tf.tanh, cnn_extractor=nature_cnn, layer_norm=False, feature_extraction="cnn",
                 **kwargs):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(LstmPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm,), reuse=reuse,
                                         scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if net_arch is None:  # Legacy mode
            if layers is None:
                layers = [64, 64]
            else:
                warnings.warn("The layers parameter is deprecated. Use the net_arch parameter instead.")

            with tf.variable_scope("model", reuse=reuse):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    for i, layer_size in enumerate(layers):
                        extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                            init_scale=np.sqrt(2)))
                input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
                masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=layer_norm)
                rnn_output = seq_to_batch(rnn_output)
                value_fn = linear(rnn_output, 'vf', 1)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

            self._value_fn = value_fn
        else:  # Use the new net_arch parameter
            if layers is not None:
                warnings.warn("The new net_arch parameter overrides the deprecated layers parameter.")
            if feature_extraction == "cnn":
                raise NotImplementedError()

            with tf.variable_scope("model", reuse=reuse):
                latent = tf.layers.flatten(self.processed_obs)
                policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
                value_only_layers = []  # Layer sizes of the network that only belongs to the value network

                # Iterate through the shared layers and build the shared parts of the network
                lstm_layer_constructed = False
                for idx, layer in enumerate(net_arch):
                    if isinstance(layer, int):  # Check that this is a shared layer
                        layer_size = layer
                        latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
                    elif layer == "lstm":
                        if lstm_layer_constructed:
                            raise ValueError("The net_arch parameter must only contain one occurrence of 'lstm'!")
                        input_sequence = batch_to_seq(latent, self.n_env, n_steps)
                        masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                        rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                                     layer_norm=layer_norm)
                        latent = seq_to_batch(rnn_output)
                        lstm_layer_constructed = True
                    else:
                        assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                        if 'pi' in layer:
                            assert isinstance(layer['pi'],
                                              list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                            policy_only_layers = layer['pi']

                        if 'vf' in layer:
                            assert isinstance(layer['vf'],
                                              list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                            value_only_layers = layer['vf']
                        break  # From here on the network splits up in policy and value network

                # Build the non-shared part of the policy-network
                latent_policy = latent
                for idx, pi_layer_size in enumerate(policy_only_layers):
                    if pi_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the policy network.")
                    assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                    latent_policy = act_fun(
                        linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

                # Build the non-shared part of the value-network
                latent_value = latent
                for idx, vf_layer_size in enumerate(value_only_layers):
                    if vf_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the value function "
                                                  "network.")
                    assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                    latent_value = act_fun(
                        linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

                if not lstm_layer_constructed:
                    raise ValueError("The net_arch parameter must contain at least one occurrence of 'lstm'!")

                self._value_fn = linear(latent_value, 'vf', 1)
                # TODO: why not init_scale = 0.001 here like in the feedforward
                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})


class CnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="cnn", **_kwargs)


class CnnMlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(CnnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                           cnn_extractor=cnn_1d_extractor, feature_extraction="cnn", **_kwargs)


class LnCnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(LnCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="cnn", layer_norm=True, **_kwargs)


class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)


class LnMlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, **_kwargs):
        super(LnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                          feature_extraction="mlp", layer_norm=True, **_kwargs)


register_policy("LstmFFMlpPolicy", LstmFFMlpPolicy)
register_policy("LstmMlpPolicy", LstmMlpPolicy)
register_policy("DRPolicy", DRPolicy)
register_policy("CnnPolicy", CnnPolicy)
register_policy("LnCnnPolicy", LnCnnPolicy)
register_policy("MlpPolicy", MlpPolicy)
register_policy("LnMlpPolicy", LnMlpPolicy)
register_policy("CnnMlpPolicy", CnnMlpPolicy)
