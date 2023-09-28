"""Wrapper that autoreset environments when `terminated=True` or `truncated=True`."""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Callable

import gymnasium as gym


if TYPE_CHECKING:
    from gymnasium.envs.registration import EnvSpec


import numpy as np



class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class MultiAgentAutoResetWrapper(gym.Wrapper):
    """A class for providing an automatic reset functionality for gymnasium environments when calling :meth:`self.step`.

    When calling step causes :meth:`Env.step` to return `terminated=True` or `truncated=True`, :meth:`Env.reset` is called,
    and the return format of :meth:`self.step` is as follows: ``(new_obs, final_reward, final_terminated, final_truncated, info)``
    with new step API and ``(new_obs, final_reward, final_done, info)`` with the old step API.

     - ``new_obs`` is the first observation after calling :meth:`self.env.reset`
     - ``final_reward`` is the reward after calling :meth:`self.env.step`, prior to calling :meth:`self.env.reset`.
     - ``final_terminated`` is the terminated value before calling :meth:`self.env.reset`.
     - ``final_truncated`` is the truncated value before calling :meth:`self.env.reset`. Both `final_terminated` and `final_truncated` cannot be False.
     - ``info`` is a dict containing all the keys from the info dict returned by the call to :meth:`self.env.reset`,
       with an additional key "final_observation" containing the observation returned by the last call to :meth:`self.env.step`
       and "final_info" containing the info dict returned by the last call to :meth:`self.env.step`.

    Warning:
        When using this wrapper to collect rollouts, note that when :meth:`Env.step` returns `terminated` or `truncated`, a
        new observation from after calling :meth:`Env.reset` is returned by :meth:`Env.step` alongside the
        final reward, terminated and truncated state from the previous episode.
        If you need the final state from the previous episode, you need to retrieve it via the
        "final_observation" key in the info dict.
        Make sure you know what you're doing if you use this wrapper!
    """

    def __init__(self, env: gym.Env):
        """A class for providing an automatic reset functionality for gymnasium environments when calling :meth:`self.step`.

        Args:
            env (gym.Env): The environment to apply the wrapper
        """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        """Steps through the environment with action and resets the environment if a terminated or truncated signal is encountered.

        Args:
            action: The action to take

        Returns:
            The autoreset environment :meth:`step`
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if all(terminated) or truncated:
            new_obs, new_info = self.env.reset()
            assert (
                "final_observation" not in new_info
            ), 'info dict cannot contain key "final_observation" '
            assert (
                "final_info" not in new_info
            ), 'info dict cannot contain key "final_info" '

            new_info["final_observation"] = obs
            new_info["final_info"] = info

            obs = new_obs
            info = new_info

        return obs, reward, terminated, truncated, info

    @property
    def spec(self) -> EnvSpec | None:
        """Modifies the environment spec to specify the `autoreset=True`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            env_spec = deepcopy(env_spec)
            env_spec.autoreset = True

        self._cached_spec = env_spec
        return env_spec
    

class AutoResetWrapper(gym.Wrapper):
    """A class for providing an automatic reset functionality for gymnasium environments when calling :meth:`self.step`.

    When calling step causes :meth:`Env.step` to return `terminated=True` or `truncated=True`, :meth:`Env.reset` is called,
    and the return format of :meth:`self.step` is as follows: ``(new_obs, final_reward, final_terminated, final_truncated, info)``
    with new step API and ``(new_obs, final_reward, final_done, info)`` with the old step API.

     - ``new_obs`` is the first observation after calling :meth:`self.env.reset`
     - ``final_reward`` is the reward after calling :meth:`self.env.step`, prior to calling :meth:`self.env.reset`.
     - ``final_terminated`` is the terminated value before calling :meth:`self.env.reset`.
     - ``final_truncated`` is the truncated value before calling :meth:`self.env.reset`. Both `final_terminated` and `final_truncated` cannot be False.
     - ``info`` is a dict containing all the keys from the info dict returned by the call to :meth:`self.env.reset`,
       with an additional key "final_observation" containing the observation returned by the last call to :meth:`self.env.step`
       and "final_info" containing the info dict returned by the last call to :meth:`self.env.step`.

    Warning:
        When using this wrapper to collect rollouts, note that when :meth:`Env.step` returns `terminated` or `truncated`, a
        new observation from after calling :meth:`Env.reset` is returned by :meth:`Env.step` alongside the
        final reward, terminated and truncated state from the previous episode.
        If you need the final state from the previous episode, you need to retrieve it via the
        "final_observation" key in the info dict.
        Make sure you know what you're doing if you use this wrapper!
    """

    def __init__(self, env: gym.Env):
        """A class for providing an automatic reset functionality for gymnasium environments when calling :meth:`self.step`.

        Args:
            env (gym.Env): The environment to apply the wrapper
        """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        """Steps through the environment with action and resets the environment if a terminated or truncated signal is encountered.

        Args:
            action: The action to take

        Returns:
            The autoreset environment :meth:`step`
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            new_obs, new_info = self.env.reset()
            assert (
                "final_observation" not in new_info
            ), 'info dict cannot contain key "final_observation" '
            assert (
                "final_info" not in new_info
            ), 'info dict cannot contain key "final_info" '

            new_info["final_observation"] = obs
            new_info["final_info"] = info

            obs = new_obs
            info = new_info

        return obs, reward, terminated, truncated, info

    @property
    def spec(self) -> EnvSpec | None:
        """Modifies the environment spec to specify the `autoreset=True`."""
        if self._cached_spec is not None:
            return self._cached_spec

        env_spec = self.env.spec
        if env_spec is not None:
            env_spec = deepcopy(env_spec)
            env_spec.autoreset = True

        self._cached_spec = env_spec
        return env_spec
    


class NormalizeReward(gym.core.Wrapper):
    """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        # gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma * (1 - terminateds) + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)
    

class MultiAgentNormalizeReward(gym.core.Wrapper):
    """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        num_agents: int,
        gamma: float = 0.99,
        epsilon: float = 1e-8
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        # gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        # self.num_envs = getattr(env, "num_envs", 1)
        # self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(num_agents)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        terminateds = np.array(terminateds)
        # if not self.is_vector_env:
        rews = np.array(rews)
        self.returns = self.returns * self.gamma * (1 - terminateds) + rews
        rews = self.normalize(rews)
        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)
    

class TransformReward(gym.RewardWrapper):
    """Transform the reward via an arbitrary function.

    Warning:
        If the base environment specifies a reward range which is not invariant under :attr:`f`, the :attr:`reward_range` of the wrapped environment will be incorrect.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformReward
        >>> env = gym.make("CartPole-v1")
        >>> env = TransformReward(env, lambda r: 0.01*r)
        >>> _ = env.reset()
        >>> observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        >>> reward
        0.01
    """

    def __init__(self, env: gym.Env, f: Callable[[float], float]):
        """Initialize the :class:`TransformReward` wrapper with an environment and reward transform function :attr:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the reward
        """
        gym.RewardWrapper.__init__(self, env)

        assert callable(f)
        self.f = f

    def reward(self, reward):
        """Transforms the reward using callable :attr:`f`.

        Args:
            reward: The reward to transform

        Returns:
            The transformed reward
        """
        return self.f(reward)
    


