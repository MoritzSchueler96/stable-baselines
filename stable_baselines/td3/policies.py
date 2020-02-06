import tensorflow as tf
import numpy as np
from gym.spaces import Box
import copy

from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy, cnn_1d_extractor
from stable_baselines.sac.policies import mlp
from stable_baselines.a2c.utils import lstm, batch_to_seq, seq_to_batch
from larnn import LinearAntisymmetricCell
from larnn import utils


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
        assert (np.abs(ac_space.low) == ac_space.high).all(), "Error: the action space low and high must be symmetric"

        self.qf1 = None
        self.qf2 = None
        self.q_discrepancy = None
        self.policy = None

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        """
        Creates an actor object

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param reuse: (bool) whether or not to resue parameters
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
        :param reuse: (bool) whether or not to resue parameters
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
    
    def __init__(self, sess, ob_space, ac_space, layers, n_env=1, n_steps=1, n_batch=None, add_state_phs=None, reuse=False,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_rnn=128, share_rnn=False, save_state=False,
                 save_target_state=False, rnn_type="lstm", layer_norm=False, act_fun=tf.nn.relu, print_tensors=False,
                 obs_module_indices=None, **kwargs):

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
        self.rnn_type = rnn_type
        self.keras_reuse = False
        self._rnn_layer = {"pi": None, "qf1": None, "qf2": None}

        self.print_tensors = print_tensors

        self.activ_fn = act_fun
        self.n_rnn = n_rnn
        self.share_rnn = share_rnn
        self._obs_ph = self.processed_obs
        self.obs_tp1_ph = self.processed_obs

        #assert self.n_batch % self.n_steps == 0, "The batch size must be a multiple of sequence length (n_steps)"
        #self._rnn_n_batch = self.n_batch // self.n_steps
        _state_shape = self.n_rnn * 2 if self.rnn_type == "lstm" else self.n_rnn

        self._initial_state = np.zeros((1, _state_shape), dtype=np.float32)

        self.action_prev = np.zeros((1, *self.ac_space.shape))

        if self.share_rnn:
            self.state = None
        else:
            self.pi_state = None
            self.qf1_state = None
            self.qf2_state = None

        with tf.variable_scope("input", reuse=False):
            self.dones_ph = tf.placeholder(tf.float32, (self.n_batch,), name="dones_ph")  # (done t-1)
            if self.share_rnn:
                self.state_ph = tf.placeholder(tf.float32, (self.n_batch, _state_shape), name="state_ph")
            else:
                self.pi_state_ph = tf.placeholder(tf.float32, (self.n_batch, _state_shape), name="pi_state_ph")
                self.qf1_state_ph = tf.placeholder(tf.float32, (self.n_batch, _state_shape), name="qf1_state_ph")
                self.qf2_state_ph = tf.placeholder(tf.float32, (self.n_batch, _state_shape), name="qf2_state_ph")

            self.action_prev_ph = tf.placeholder(np.float32, (self.n_batch, *self.ac_space.shape), name="action_prev_ph")

        self.save_state = save_state
        self.save_target_state = save_target_state

        self.extra_phs = ["action_prev", "target_action_prev"]
        self.rnn_inputs = ["obs", "action_prev"]  # TODO: rename to scan data?
        self.extra_data_names = ["action_prev", "target_action_prev"]

        if self.save_target_state:
            self.rnn_inputs = sorted(self.rnn_inputs + ["obs_tp1", "target_action_prev"])

        if self.save_state:
            state_names = ["state"] if self.share_rnn else ["pi_state", "qf1_state", "qf2_state"]
            if self.save_target_state:
                state_names.extend(["target_" + state_name for state_name in state_names])
            if self.share_rnn:
                self.extra_data_names = sorted(self.extra_data_names + state_names)
            else:
                self.extra_data_names = sorted(self.extra_data_names + state_names)

        if self.save_state or add_state_phs is not None or self.save_target_state:
            phs_to_add = []
            if self.save_state or add_state_phs in ["main", "both"]:
                phs_to_add.extend(["pi_state", "qf1_state", "qf2_state"])
            if self.save_target_state or add_state_phs in ["target", "both"]:
                phs_to_add.extend(["target_pi_state", "target_qf1_state", "target_qf2_state"])

            self.extra_phs = sorted(self.extra_phs + phs_to_add)

    def _process_phs(self, **phs):
        for ph_name, ph_val in phs.items():
            if ph_val is None:
                phs[ph_name] = getattr(self, ph_name + "_ph")
            else:
                try:
                    setattr(self, ph_name + "_ph", ph_val)
                except AttributeError:
                    setattr(self, "_" + ph_name + "_ph", ph_val)

            if self.print_tensors:
                phs[ph_name] = tf.Print(phs[ph_name], [phs[ph_name]], "{}: ".format(ph_name))

        return phs.values()

    def _make_branch(self, branch_name, input_tensor, dones=None, state_ph=None):
        if branch_name == "rnn":
            for i, fc_layer_units in enumerate(self.layers["rnn"]):
                input_tensor = self.activ_fn(tf.layers.dense(input_tensor, fc_layer_units, name="rnn_fc{}".format(i)))

            if self.rnn_type == "lstm":
                input_tensor = batch_to_seq(input_tensor, self._rnn_n_batch, self.n_steps)
                masks = batch_to_seq(dones, self._rnn_n_batch, self.n_steps)
                input_tensor, state = lstm(input_tensor, masks, state_ph, "rnn", n_hidden=self.n_rnn,
                                           layer_norm=self.layer_norm)
                input_tensor = seq_to_batch(input_tensor)
            elif self.rnn_type == "larnn":
                n_features = input_tensor.shape[-1].value
                var_scope_name = tf.get_variable_scope().original_name_scope.split("/")[-2].split("_")[0]
                if self._rnn_layer[var_scope_name] is None:
                    larnn_cell = LinearAntisymmetricCell(self.n_rnn, step_size=self._rnn_epsilon, method=self._rnn_method)
                    larnn_layer = tf.keras.layers.RNN(larnn_cell, return_state=True, return_sequences=True,
                                                      input_shape=(None, self.n_steps, n_features))
                    self._rnn_layer[var_scope_name] = larnn_layer
                else:
                    larnn_layer = self._rnn_layer[var_scope_name]
                input_tensor = tf.reshape(input_tensor, (-1, self.n_steps, n_features))
                dones = dones[::self.n_steps]
                if self.print_tensors:
                    state_ph = tf.Print(state_ph, [state_ph], "State in: ")
                #state_ph = tf.where(tf.cast(dones, dtype=np.bool),
                #                    tf.keras.backend.repeat_elements(self.initial_state, state_ph.shape[0].value, axis=0),
                #                    state_ph)  # TODO: implement done functionality
                input_tensor, state = larnn_layer(input_tensor, initial_state=state_ph)
                if self.print_tensors:
                    state = tf.Print(state, [state], "State out: ")
                input_tensor = tf.reshape(input_tensor, (-1, self.n_rnn))
            else:
                raise ValueError

            return input_tensor, state
        else:
            for i, fc_layer_units in enumerate(self.layers[branch_name]):
                input_tensor = self.activ_fn(tf.layers.dense(input_tensor, fc_layer_units, name="{}_fc{}".format(branch_name, i)))

            return input_tensor

    def make_actor(self, ff_phs=None, rnn_phs=None, dones=None, reuse=False, scope="pi"):
        rnn_branch = tf.concat([tf.layers.flatten(ph) for ph in rnn_phs], axis=-1)
        if ff_phs is not None:
            ff_branch = tf.concat([tf.layers.flatten(ph) for ph in ff_phs], axis=-1)

        if dones is None:
            dones = self.dones_ph

        if self.share_rnn:
            with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
                rnn_branch, self.state = self._make_branch("rnn", rnn_branch, dones, self.state_ph)

        with tf.variable_scope(scope, reuse=reuse):
            if self.layers["ff"] is not None:
               ff_branch = self._make_branch("ff", ff_branch)

            if not self.share_rnn:
                rnn_branch, self.pi_state = self._make_branch("rnn", rnn_branch, dones, self.pi_state_ph)

            if ff_phs is not None:
                head = tf.concat([ff_branch, rnn_branch], axis=-1)
            else:
                head = rnn_branch

            head = self._make_branch("head", head)

            self.policy_pre_activation = tf.layers.dense(head, self.ac_space.shape[0], name="pi_pre_activation")
            self.policy = policy = tf.tanh(self.policy_pre_activation, name="pi_activation")

        return policy

    def make_critics(self, ff_phs=None, rnn_phs=None, dones=None, reuse=False, scope="values_fn"):
        rnn_branch_in = tf.concat([tf.layers.flatten(ph) for ph in rnn_phs], axis=-1)
        if ff_phs is not None:
            ff_branch_in = tf.concat([tf.layers.flatten(ph) for ph in ff_phs], axis=-1)

        if dones is None:
            dones = self.dones_ph

        self.qf1, self.qf2 = None, None
        self.qf1_state, self.qf2_state = None, None

        if self.share_rnn:
            with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
                rnn_branch_s, self.state = self._make_branch("rnn", rnn_branch_in, dones, self.state_ph)

        with tf.variable_scope(scope, reuse=reuse):
            # Double Q values to reduce overestimation
            for qf_i in range(1, 3):
                with tf.variable_scope('qf{}'.format(qf_i), reuse=reuse):
                    rnn_branch = rnn_branch_in
                    if self.layers["ff"] is not None:
                        ff_branch = self._make_branch("ff", ff_branch_in)
                    elif ff_phs is not None:
                        ff_branch = ff_branch_in

                    if not self.share_rnn:
                        rnn_branch, state = self._make_branch("rnn", rnn_branch, dones,
                                                               getattr(self, "qf{}_state_ph".format(qf_i)))
                        setattr(self, "qf{}_state".format(qf_i), state)
                    else:
                        rnn_branch = rnn_branch_s

                    if ff_phs is not None:
                        head = tf.concat([ff_branch, rnn_branch], axis=-1)
                    else:
                        head = rnn_branch

                    head = self._make_branch("head", head)

                    setattr(self, "qf{}".format(qf_i), tf.layers.dense(head, 1, name="qf{}".format(qf_i)))

        return self.qf1, self.qf2

    def step(self, obs, action_prev=None, state=None, mask=None, feed_dict=None, **kwargs):
        if feed_dict is None:
            feed_dict = {}
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = np.array([False])
        if action_prev is None:
            assert obs.shape[0] == 1
            if mask[0]:
                self.action_prev = np.zeros((1, *self.ac_space.shape))
            action_prev = self.action_prev

        rnn_node = self.state if self.share_rnn else self.pi_state
        state_ph = self.state_ph if self.share_rnn else self.pi_state_ph

        feed_dict.update({self.obs_ph: obs, state_ph: state, self.dones_ph: mask,
                                      self.action_prev_ph: action_prev})

        action, out_state = self.sess.run([self.policy, rnn_node], feed_dict)
        self.action_prev = action

        return action, out_state

    @property
    def initial_state(self):
        return self._initial_state

    def collect_data(self, _locals, _globals):
        data = {}
        if self.save_state:
            if self.share_rnn:
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
                                         for data_name in self.rnn_inputs if "target" not in data_name})
                    qf1_state, qf2_state = self.sess.run([self.qf1_state, self.qf2_state], feed_dict=qf_feed_dict)
                data["qf1_state"] = qf1_state[0, :]
                data["qf2_state"] = qf2_state[0, :]

        if len(_locals["episode_data"]) == 0:
            data["action_prev"] = np.zeros(*self.ac_space.shape, dtype=np.float32)
        else:
            data["action_prev"] = _locals["episode_data"][-1]["action"]

        data["target_action_prev"] = _locals["action"]

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
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_rnn=128, share_rnn=False,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        if layers is None:
            layers = {"ff": [128], "rnn": [128], "head": [128, 128]}
        super().__init__(sess, ob_space, ac_space, layers, n_env, n_steps, n_batch,
                                                reuse=reuse, cnn_extractor=cnn_extractor,
                                                feature_extraction=feature_extraction, n_rnn=n_rnn,
                                                share_rnn=share_rnn, layer_norm=layer_norm, act_fun=act_fun,
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

    def collect_data(self, _locals, _globals, **kwargs):
        data = super().collect_data(_locals, _globals)
        if "my" not in _locals or _locals["ep_data"]:
            data["my"] = _locals["self"].env.get_env_parameters()

        return data


class LstmMlpPolicy(RecurrentPolicy):
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False,
                 layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_rnn=128, share_rnn=False,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        if layers is None:
            layers = {"ff": None, "rnn": [64, 64], "head": []}

        else:
            assert layers["ff"] is None
        super().__init__(sess, ob_space, ac_space, layers, n_env, n_steps, n_batch,
                         reuse=reuse, cnn_extractor=cnn_extractor,
                         feature_extraction=feature_extraction, n_rnn=n_rnn,
                         share_rnn=share_rnn, layer_norm=layer_norm, act_fun=act_fun,
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


class LstmFFMlpPolicy(RecurrentPolicy):
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False,
                 layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="mlp", n_rnn=128, share_rnn=False,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, **kwargs):
        if layers is None:
            layers = {"ff": [64], "lstm": [64, 64], "head": []}

        super().__init__(sess, ob_space, ac_space, layers, n_env, n_steps, n_batch,
                         reuse=reuse, cnn_extractor=cnn_extractor,
                         feature_extraction=feature_extraction, n_rnn=n_rnn,
                         share_rnn=share_rnn, layer_norm=layer_norm, act_fun=act_fun,
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


class LarnnMlpPolicy(RecurrentPolicy):
    recurrent = True

    def __init__(self, sess, ob_space, ac_space, rnn_epsilon, n_env=1, n_steps=1, n_batch=None, reuse=False,
                 layers=None, cnn_extractor=nature_cnn, feature_extraction="mlp", n_rnn=128, share_rnn=False,
                 layer_norm=False, act_fun=tf.nn.relu, obs_module_indices=None, rnn_method="midpoint", **kwargs):
        if layers is None:
            layers = {"ff": None, "rnn": [64, 64], "head": []}
        else:
            assert layers["ff"] is None
        super().__init__(sess, ob_space, ac_space, layers, n_env, n_steps, n_batch,
                         reuse=reuse, cnn_extractor=cnn_extractor,
                         feature_extraction=feature_extraction, n_rnn=n_rnn, rnn_type="larnn",
                         share_rnn=share_rnn, layer_norm=layer_norm, act_fun=act_fun,
                         obs_module_indices=obs_module_indices, **kwargs)

        self.keras_reuse = True
        self._rnn_epsilon = rnn_epsilon
        self._rnn_method = rnn_method

    def make_actor(self, obs=None, action_prev=None, dones=None, reuse=False, scope="pi"):
        obs, action_prev, dones = self._process_phs(obs=obs, action_prev=action_prev, dones=dones)

        ff_phs = None
        rnn_phs = [obs, action_prev]
        return super().make_actor(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)

    def make_critics(self, obs=None, action=None, action_prev=None, dones=None, reuse=False, scope="values_fn"):
        obs, action, action_prev, dones = self._process_phs(obs=obs, action=action, action_prev=action_prev,
                                                            dones=dones)

        ff_phs = [action]
        rnn_phs = [obs, action_prev]
        return super().make_critics(ff_phs=ff_phs, rnn_phs=rnn_phs, dones=dones, reuse=reuse, scope=scope)


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
register_policy("LarnnMlpPolicy", LarnnMlpPolicy)
