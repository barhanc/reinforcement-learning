from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Literal

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .utils import combined_shape


class BaseBuffer(ABC):
    @abstractmethod
    def __init__(self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None) -> None:
        self.device = device

        self.actions = torch.zeros(
            combined_shape(size, env.action_space.shape),
            dtype=torch.float32,
            device=device,
        )
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.terminations = torch.zeros(size, dtype=torch.float32, device=device)
        self.truncations = torch.zeros(size, dtype=torch.float32, device=device)
        self.infos = np.empty((size,), dtype=object)
        self._ptr, self.size, self.max_size = 0, 0, size

    def store(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        action: NDArray,
        reward: float,
        next_observation: Union[NDArray, dict[str, NDArray]],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        self._store_observations(observation, next_observation)
        self.actions[self._ptr] = torch.as_tensor(action, dtype=torch.float32)
        self.rewards[self._ptr] = torch.as_tensor(reward, dtype=torch.float32)
        self.terminations[self._ptr] = torch.as_tensor(terminated, dtype=torch.float32)
        self.truncations[self._ptr] = torch.as_tensor(truncated, dtype=torch.float32)
        self.infos[self._ptr] = info
        self._ptr = (self._ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    @abstractmethod
    def _store_observations(
        self,
        observation: Union[NDArray, dict[str, NDArray]],
        next_observation: Union[NDArray, dict[str, NDArray]],
    ) -> None: ...

    def sample_batch(self, batch_size: int = 32) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        idxs = torch.randint(0, self.size, size=(batch_size,))
        # idxs = np.random.randint(0, self.size, size=batch_size)
        return self.batch(idxs)

    def batch(self, idxs: Tensor) -> dict[str, Union[Tensor, dict[str, Tensor]]]:
        data = dict(
            action=self.actions[idxs],
            reward=self.rewards[idxs],
            terminated=self.terminations[idxs],
            truncated=self.truncations[idxs],
            info=self.infos[idxs],
        )
        observations = self._observations_batch(idxs)
        data.update(observations)

        return data

    @abstractmethod
    def _observations_batch(self, idxs: Tensor) -> dict[str, Union[Tensor, dict[str, Tensor]]]: ...

    def start_episode(self):
        pass

    def end_episode(self):
        pass

    def clear(self):
        self.actions.zero_()
        self.rewards.zero_()
        self.terminations.zero_()
        self.truncations.zero_()
        self.infos.fill(None)
        self._ptr, self.size = 0, 0


class DictReplayBuffer(BaseBuffer):
    """
    A dictionary experience replay buffer for off-policy agents.
    """

    def __init__(self, env: gym.Env, size: int = 100000, device: Optional[torch.device] = None):
        assert isinstance(env.observation_space, gym.spaces.Dict)
        super().__init__(env=env, size=size, device=device)

        obs_space = {k: combined_shape(size, v.shape) for k, v in env.observation_space.items()}

        self.observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device) for k, v in env.observation_space.items()
        }
        self.next_observations: dict[str, Tensor] = {
            k: torch.zeros(obs_space[k], dtype=torch.float32, device=device) for k, v in env.observation_space.items()
        }

    def _store_observations(
        self,
        observation: dict[str, NDArray],
        next_observation: dict[str, NDArray],
    ) -> None:
        for k in observation.keys():
            self.observations[k][self._ptr] = torch.as_tensor(observation[k], dtype=torch.float32)
        for k in next_observation.keys():
            self.next_observations[k][self._ptr] = torch.as_tensor(next_observation[k], dtype=torch.float32)

    def _observations_batch(self, idxs: Tensor) -> dict[str, dict[str, Tensor]]:
        return dict(
            observation={k: v[idxs] for k, v in self.observations.items()},
            next_observation={k: v[idxs] for k, v in self.next_observations.items()},
        )


class HerReplayBuffer(DictReplayBuffer):
    def __init__(
        self,
        env: gym.Env,
        size: int = 100000,
        device: Optional[torch.device] = None,
        n_sampled_goal: int = 1,
        goal_selection_strategy: str = "final",
    ):
        super().__init__(env=env, size=size, device=device)
        self.env = env
        self.n_sampled_goal = n_sampled_goal
        self.selection_strategy = goal_selection_strategy
        # TODO: fill this in
        # You can put additional attributes here if needed.
        # Also: There is a number of methods in the base class that could be useful to override.
        self._start_ptr = None

    def start_episode(self):
        self._start_ptr = self._ptr

    def end_episode(self):
        if self._start_ptr <= self._ptr:
            idx = np.arange(self._start_ptr, self._ptr)
        else:
            idx = np.concat((np.arange(self._start_ptr, self.size), np.arange(self._ptr)))
        episode = self.batch(idx)
        self._augment(episode)

    def _augment(self, episode: dict[str, Union[Tensor, dict[str, Tensor]]]):
        length = len(episode["info"])

        for curr_idx in range(length):
            action = episode["action"][curr_idx]
            observation = {k: v[curr_idx] for k, v in episode["observation"].items()}
            next_observation = {k: v[curr_idx] for k, v in episode["next_observation"].items()}

            for _ in range(self.n_sampled_goal):
                match self.selection_strategy:
                    case "final":
                        next_idx = -1
                    case "future":
                        if curr_idx == length - 1:
                            break
                        next_idx = np.random.randint(curr_idx + 1, length)
                    case _:
                        raise ValueError(f"Not recognised {self.selection_strategy=}")

                reward = self.env.unwrapped.compute_reward(
                    episode["next_observation"]["achieved_goal"][curr_idx].cpu().numpy(),
                    episode["next_observation"]["achieved_goal"][next_idx].cpu().numpy(),
                    episode["info"][next_idx],
                )

                observation["desired_goal"] = episode["next_observation"]["achieved_goal"][next_idx]
                next_observation["desired_goal"] = episode["next_observation"]["achieved_goal"][next_idx]

                self.store(
                    observation=observation,
                    next_observation=next_observation,
                    action=action,
                    reward=reward,
                    info=episode["info"][next_idx],
                    truncated=episode["truncated"][next_idx],
                    terminated=episode["terminated"][next_idx],
                )

    def store(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_observation: dict[str, torch.Tensor],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ):
        # TODO: fill this in
        # Just a suggestion: it may make sense to modify this method

        # Store the transition
        super().store(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

        # TODO: fill this in
        # Or maybe here?
