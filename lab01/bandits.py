import math
import random

import numpy as np
import matplotlib.pyplot as plt

from typing import Protocol
from abc import abstractmethod
from itertools import accumulate


class KArmedBandit(Protocol):
    @abstractmethod
    def arms(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def reward(self, arm: str) -> float:
        raise NotImplementedError


class BanditLearner(Protocol):
    name: str
    color: str

    @abstractmethod
    def reset(self, arms: list[str], time_steps: int):
        raise NotImplementedError

    @abstractmethod
    def pick_arm(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def acknowledge_reward(self, arm: str, reward: float) -> None:
        pass


class BanditProblem:
    def __init__(self, time_steps: int, bandit: KArmedBandit, learner: BanditLearner):
        self.time_steps: int = time_steps
        self.bandit: KArmedBandit = bandit
        self.learner: BanditLearner = learner
        self.learner.reset(self.bandit.arms(), self.time_steps)

    def run(self) -> list[float]:
        rewards = []
        for _ in range(self.time_steps):
            arm = self.learner.pick_arm()
            reward = self.bandit.reward(arm)
            self.learner.acknowledge_reward(arm, reward)
            rewards.append(reward)
        return rewards


POTENTIAL_HITS = {
    "In Praise of Dreams": 0.8,
    "We Built This City": 0.9,
    "April Showers": 0.5,
    "Twenty Four Hours": 0.3,
    "Dirge for November": 0.1,
}


class TopHitBandit(KArmedBandit):
    def __init__(self, potential_hits: dict[str, float]):
        self.potential_hits: dict[str, float] = potential_hits

    def arms(self) -> list[str]:
        return list(self.potential_hits)

    def reward(self, arm: str) -> float:
        thumb_up_probability = self.potential_hits[arm]
        return 1.0 if random.random() <= thumb_up_probability else 0.0


class RandomLearner(BanditLearner):
    def __init__(self):
        self.name = "Random"
        self.color = "black"
        self.arms: list[str] = []

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms

    def pick_arm(self) -> str:
        return random.choice(self.arms)

    def acknowledge_reward(self, arm: str, reward: float) -> None:
        pass


class ExploreThenCommitLearner(BanditLearner):
    def __init__(self, m: int, color: str = "blue"):
        self.name = "ExploreThanCommit"
        self.color = color

        self.arms: list[str] = []
        self.m = m

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.time = 0
        self.picks = {arm: 0 for arm in arms}
        self.reward = {arm: 0 for arm in arms}

    def pick_arm(self) -> str:
        if self.time < self.m * len(self.arms):
            arm = self.arms[self.time % len(self.arms)]
        else:
            arm = max(self.reward, key=lambda arm: (self.reward[arm], random.random()))

        self.time += 1
        self.picks[arm] += 1
        return arm

    def acknowledge_reward(self, arm: str, reward: float):
        if self.time < self.m * len(self.arms):
            self.reward[arm] += 1 / self.picks[arm] * (reward - self.reward[arm])
        else:  # Don't acknowledge
            ...


class EpsilonGreedyLearner(BanditLearner):
    def __init__(self, eps: float, color="red", init_reward: float = 0.0):
        self.name = "EpsilonGreedy"
        self.color = color
        self.eps = eps
        self.init_reward = init_reward

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.picks = {arm: 0 for arm in arms}
        self.reward = {arm: self.init_reward for arm in arms}

    def pick_arm(self) -> str:
        if random.random() < self.eps:
            arm = random.choice(self.arms)
        else:
            arm = max(self.reward, key=lambda arm: (self.reward[arm], random.random()))

        self.picks[arm] += 1
        return arm

    def acknowledge_reward(self, arm: str, reward: float):
        self.reward[arm] += 1 / self.picks[arm] * (reward - self.reward[arm])


class UCB1Learner(BanditLearner):
    def __init__(self, c: float):
        self.name = "UCB1"
        self.color = "green"
        self.c = c

    def reset(self, arms: int, time_steps: int):
        self.time = 0
        self.arms = arms
        self.picks = {arm: 0 for arm in arms}
        self.reward = {arm: 0 for arm in arms}

    def u_bound(self, arm: str) -> float:
        if self.picks[arm] == 0:
            return float("inf")
        return self.c * math.sqrt(math.log(self.time) / self.picks[arm])

    def pick_arm(self) -> str:
        arm = max(self.arms, key=lambda arm: self.reward[arm] + self.u_bound(arm))
        self.time += 1
        self.picks[arm] += 1
        return arm

    def acknowledge_reward(self, arm: str, reward: float):
        self.reward[arm] += 1 / self.picks[arm] * (reward - self.reward[arm])


class ThompsonLearner(BanditLearner):
    def __init__(self):
        self.name = "Thompson"
        self.color = "orange"

    def reset(self, arms: str, time_steps: int): ...


class GradientLearner(BanditLearner):
    def __init__(self):
        self.name = "Gradient"
        self.color = "purple"

    def reset(self, arms: str, time_steps: int):
        self.reward = {arm: 0 for arm in arms}
        self.picks = {arm: 0 for arm in arms}

    def pick_arm(self) -> str: ...

    def acknowledge_reward(self, arm: str, reward: float): ...


TIME_STEPS = 1000
TRIALS_PER_LEARNER = 50


def evaluate_learner(learner: BanditLearner) -> None:
    runs_results = []
    for _ in range(TRIALS_PER_LEARNER):
        bandit = TopHitBandit(POTENTIAL_HITS)
        problem = BanditProblem(time_steps=TIME_STEPS, bandit=bandit, learner=learner)
        rewards = problem.run()
        accumulated_rewards = list(accumulate(rewards))
        runs_results.append(accumulated_rewards)

    runs_results = np.array(runs_results)
    mean_accumulated_rewards = np.mean(runs_results, axis=0)
    std_accumulated_rewards = np.std(runs_results, axis=0)
    plt.plot(mean_accumulated_rewards, label=learner.name, color=learner.color)
    plt.fill_between(
        range(len(mean_accumulated_rewards)),
        mean_accumulated_rewards - std_accumulated_rewards,
        mean_accumulated_rewards + std_accumulated_rewards,
        color=learner.color,
        alpha=0.2,
    )


def main():
    learners = [
        RandomLearner(),
        ExploreThenCommitLearner(m=15),
        EpsilonGreedyLearner(eps=0.01, init_reward=10),
        UCB1Learner(c=0.5),
    ]
    for learner in learners:
        evaluate_learner(learner)

    plt.xlabel("Czas")
    plt.ylabel("Suma uzyskanych nagród")
    plt.xlim(0, TIME_STEPS)
    plt.ylim(0, TIME_STEPS)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
