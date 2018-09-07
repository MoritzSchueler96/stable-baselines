import pytest

from stable_baselines.a2c import A2C
from stable_baselines.acer import ACER
from stable_baselines.acktr import ACKTR
from stable_baselines.deepq import DeepQ, MlpPolicy as DeepQMlpPolicy
from stable_baselines.ddpg import DDPG
from stable_baselines.ppo1 import PPO1
from stable_baselines.ppo2 import PPO2
from stable_baselines.trpo_mpi import TRPO
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import FeedForwardPolicy as DDPGFeedForwardPolicy


learn_func_list = [
    lambda e: A2C(policy=MlpPolicy, learning_rate=1e-3, n_steps=1,
                  gamma=0.7, env=e).learn(total_timesteps=10000, seed=0),
    lambda e: ACER(policy=MlpPolicy, env=e,
                   n_steps=1, replay_ratio=1).learn(total_timesteps=15000, seed=0),
    lambda e: ACKTR(policy=MlpPolicy, env=e, learning_rate=5e-4, n_steps=1).learn(total_timesteps=20000, seed=0),
    lambda e: DeepQ(policy=DeepQMlpPolicy, batch_size=16, gamma=0.1,
                    exploration_fraction=0.001, env=e).learn(total_timesteps=40000, seed=0),
    lambda e: PPO1(policy=MlpPolicy, env=e, lam=0.5,
                   optim_batchsize=16, optim_stepsize=1e-3).learn(total_timesteps=15000, seed=0),
    lambda e: PPO2(policy=MlpPolicy, env=e, learning_rate=1.5e-3,
                   lam=0.8).learn(total_timesteps=20000, seed=0),
    lambda e: TRPO(policy=MlpPolicy, env=e, max_kl=0.05, lam=0.7).learn(total_timesteps=10000, seed=0),
]


class DDPGCustomPolicy(DDPGFeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(DDPGCustomPolicy, self).__init__(*args, **kwargs,
                                               layers=[64],
                                               feature_extraction="mlp")


# @pytest.mark.slow
# @pytest.mark.parametrize("learn_func", learn_func_list)
# def test_identity(learn_func):
#     """
#     Test if the algorithm (with a given policy)
#     can learn an identity transformation (i.e. return observation as an action)
#
#     :param learn_func: (lambda (Gym Environment): A2CPolicy) the policy generator
#     """
#     env = DummyVecEnv([lambda: IdentityEnv(10)])
#
#     model = learn_func(env)
#
#     n_trials = 1000
#     reward_sum = 0
#     obs = env.reset()
#     for _ in range(n_trials):
#         action, _ = model.predict(obs)
#         obs, reward, _, _ = env.step(action)
#         reward_sum += reward
#     assert reward_sum > 0.9 * n_trials
#     # Free memory
#     del model, env


@pytest.mark.slow
def test_identity_ddpg():
    """
    Test if the algorithm (with a given policy)
    can learn an identity transformation (i.e. return observation as an action)
    """
    env = DummyVecEnv([lambda: IdentityEnvBox(eps=0.5)])

    import numpy as np
    from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    std = 0.2
    param_noise = None
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(std), desired_action_stddev=float(std))
    action_noise = None
    # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(std) * np.ones(n_actions))

    # FIXME: this test fail for now
    model = DDPG(DDPGCustomPolicy, env, enable_popart=False, gamma=0.99, actor_lr=1e-4, critic_lr=1e-3, batch_size=64,
                 layer_norm=True, normalize_observations=True, normalize_returns=False, critic_l2_reg=1e-2, reward_scale=1.,
                 param_noise=param_noise, action_noise=action_noise, tensorboard_log="/tmp/dddpg/").learn(total_timesteps=50000, seed=0)

    std = 0.2
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(std), desired_action_stddev=float(std))
    # model = DDPG(DDPGMlpPolicy, env, nb_train_steps=4, nb_rollout_steps=4, param_noise=param_noise, gamma=0.5,
    #              normalize_returns=True, tau=0.001, batch_size=16, tensorboard_log="/tmp/dddpg/").learn(total_timesteps=20000, seed=0)

    n_trials = 1000
    reward_sum = 0
    obs = env.reset()
    for _ in range(n_trials):
        action, _ = model.predict(obs)
        obs, reward, _, _ = env.step(action)
        reward_sum += reward
    assert reward_sum > 0.9 * n_trials
    # Free memory
    del model, env
