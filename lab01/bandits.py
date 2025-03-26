import math
import random

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from numpy.typing import NDArray

from itertools import cycle
from abc import abstractmethod
from collections import defaultdict
from typing import Protocol, Optional, Sequence


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


class TopHitBandit(KArmedBandit):
    def __init__(self, potential_hits: dict[str, float]):
        self.potential_hits: dict[str, float] = potential_hits

    def arms(self) -> list[str]:
        return list(self.potential_hits)

    def reward(self, arm: str) -> float:
        thumb_up_probability = self.potential_hits[arm]
        return 1.0 if random.random() <= thumb_up_probability else 0.0


class DriftingTopHitBandit(KArmedBandit):
    def __init__(self, potential_hits: list[dict[str, float]]):
        self._arms = list(potential_hits[0])
        self.potential_hits = cycle(potential_hits)

    def arms(self) -> list[str]:
        return self._arms

    def reward(self, arm: str) -> float:
        thumb_up_probability = next(self.potential_hits)[arm]
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

    def acknowledge_reward(self, arm: str, reward: float) -> None: ...


class ExploreThenCommitLearner(BanditLearner):
    def __init__(self, m: int, color: str = "blue"):
        self.name = f"ExploreThenCommit $m=${m}"
        self.color = color
        self.arms: list[str] = []
        # Params
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
            pass


class EpsilonGreedyLearner(BanditLearner):
    def __init__(
        self,
        eps: float,
        color="red",
        init_reward: float = 0.0,
        alpha: Optional[float] = None,
    ):
        self.name = f"EpsilonGreedy ϵ={eps}, $Q_0$={init_reward}"
        self.color = color
        self.arms: list[str] = []
        # Params
        self.eps = eps
        self.alpha = alpha
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
        alpha = 1 / self.picks[arm] if self.alpha is None else self.alpha
        self.reward[arm] += alpha * (reward - self.reward[arm])


class UCB1Learner(BanditLearner):
    def __init__(self, c: float):
        self.name = f"UCB1 $c=${c}"
        self.color = "green"
        self.arms: list[str] = []
        # Params
        self.c = c

    def reset(self, arms: list[str], time_steps: int):
        self.time = 0
        self.arms = arms
        self.picks = {arm: 0 for arm in arms}
        self.reward = {arm: 0 for arm in arms}

    def _bound(self, arm: str) -> float:
        if self.picks[arm] == 0:
            return float("inf")
        return self.c * math.sqrt(math.log(self.time) / self.picks[arm])

    def pick_arm(self) -> str:
        arm = max(self.arms, key=lambda arm: self.reward[arm] + self._bound(arm))
        self.time += 1
        self.picks[arm] += 1
        return arm

    def acknowledge_reward(self, arm: str, reward: float):
        self.reward[arm] += 1 / self.picks[arm] * (reward - self.reward[arm])


def softmax(x: NDArray) -> NDArray:
    m = x.max()
    return np.exp(x - m) / np.sum(np.exp(x - m))


class GradientLearner(BanditLearner):
    def __init__(self, lr: float):
        self.name = f"Gradient α={lr}"
        self.color = "brown"
        self.arms: list[str] = []
        # Params
        self.lr = lr

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.picks = 0
        self.reward = 0.0
        self.H = {arm: 0.0 for arm in arms}

    def pick_arm(self) -> str:
        self.picks += 1
        h = np.array(list(self.H.values()))
        i = np.random.multinomial(n=1, pvals=softmax(h))
        return self.arms[np.where(i == 1)[0][0]]

    def acknowledge_reward(self, arm: str, reward: float):
        for i, a in enumerate(self.arms):
            h = np.array(list(self.H.values()))
            self.H[a] += self.lr * (reward - self.reward) * ((a == arm) - softmax(h)[i])
        self.reward += 1 / self.picks * (reward - self.reward)


class ThompsonLearner(BanditLearner):
    def __init__(self, a0: float = 1.0, b0: float = 1.0):
        self.name = f"Thompson α0={a0}, β0={b0}"
        self.color = "orange"
        self.arms: list[str] = []
        # Params
        self.a0 = a0
        self.b0 = b0

    def reset(self, arms: list[str], time_steps: int):
        self.arms = arms
        self.a = {arm: self.a0 for arm in arms}
        self.b = {arm: self.b0 for arm in arms}

    def pick_arm(self) -> str:
        proba = {arm: np.random.beta(self.a[arm], self.b[arm]) for arm in self.arms}
        return max(self.arms, key=lambda arm: proba[arm])

    def acknowledge_reward(self, arm: str, reward: float):
        self.a[arm] += reward
        self.b[arm] += 1 - reward


def evaluate_learner(
    learner: BanditLearner,
    potential_hits: dict[str, float] | Sequence[dict[str, float]],
    time_steps: int,
    trials_per_learner: int,
    bandit_class: type[KArmedBandit] = TopHitBandit,
    **bandit_kwargs,
) -> NDArray:

    runs_rewards = []

    for _ in range(trials_per_learner):
        bandit = bandit_class(potential_hits, **bandit_kwargs)
        problem = BanditProblem(time_steps=time_steps, bandit=bandit, learner=learner)
        rewards = problem.run()
        runs_rewards.append(rewards)

    return np.array(runs_rewards)


def compute_avg_regret(
    learner: BanditLearner,
    potential_hits_list: Sequence[dict[str, float]],
    time_steps: int,
    trials_per_learner: int,
) -> NDArray:

    avg_regret = 0

    for potential_hits in potential_hits_list:
        runs_rewards = evaluate_learner(learner, potential_hits, time_steps, trials_per_learner)
        mean = runs_rewards.cumsum(axis=1).mean(axis=0)
        avg_regret += np.arange(time_steps) * max(potential_hits.values()) - mean

    avg_regret /= len(potential_hits_list)
    return avg_regret


def task1():
    """
    Podstawowy eksperyment
    ----------------------
    Podstawowy eksperyment to uruchomienie istniejącego już szkieletu kodu i badanie, jak będą
    kumulowały się w czasie nagrody otrzymane przez poszczególne algorytmy.
    """
    TIME_STEPS = 1_000
    TRIALS_PER_LEARNER = 50
    POTENTIAL_HITS = {
        "In Praise of Dreams": 0.8,
        "We Built This City": 0.9,
        "April Showers": 0.5,
        "Twenty Four Hours": 0.3,
        "Dirge for November": 0.1,
    }

    learners: list[BanditLearner] = [
        RandomLearner(),
        ExploreThenCommitLearner(m=50),
        EpsilonGreedyLearner(eps=0.1, init_reward=0),
        UCB1Learner(c=0.1),
        GradientLearner(lr=1.0),
        ThompsonLearner(a0=1, b0=1),
    ]

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots(nrows=2, ncols=3)

    fig1.tight_layout()
    fig2.tight_layout()

    for i, learner in enumerate(learners):
        runs_rewards = evaluate_learner(learner, POTENTIAL_HITS, TIME_STEPS, TRIALS_PER_LEARNER)
        mean = runs_rewards.cumsum(axis=1).mean(axis=0)
        std = runs_rewards.cumsum(axis=1).std(axis=0)

        ax1.plot(mean, lw=0.8, c=learner.color, label=learner.name)
        ax2[i // 3, i % 3].fill_between(
            range(len(mean)),
            mean - std,
            mean + std,
            color=learner.color,
            label=learner.name,
            alpha=0.2,
        )
        ax2[i // 3, i % 3].legend(loc="upper left")
        ax2[i // 3, i % 3].set_xlim(0, TIME_STEPS)
        ax2[i // 3, i % 3].set_ylim(0, TIME_STEPS)
        ax2[i // 3, i % 3].set_xlabel("Time")
        ax2[i // 3, i % 3].set_ylabel(f"Cumulative reward std.\n({TRIALS_PER_LEARNER} trials)")

    ax1.legend(loc="best")
    ax1.set_xlim(0, TIME_STEPS)
    ax1.set_ylim(0, TIME_STEPS)
    ax1.set_xlabel("Time")
    ax1.set_ylabel(f"Cumulative reward mean ({TRIALS_PER_LEARNER} trials)")

    plt.show()


def task2():
    """
    Studium parametryczne
    ---------------------
    Jeżeli algorytm jest zależny od pewnych meta-parametrów, to warto ocenić go w pełnym przekroju
    ich wartości (patrz odpowiedni slajd na wykładzie). Dla zaimplementowanych przez nas rozwiązań
    warto przygotować analogiczne studium.
    """
    TIME_STEPS = 1_000
    TRIALS_PER_LEARNER = 50
    POTENTIAL_HITS = {
        "In Praise of Dreams": 0.8,
        "We Built This City": 0.9,
        "April Showers": 0.5,
        "Twenty Four Hours": 0.3,
        "Dirge for November": 0.1,
    }

    study = {
        "ExploreThenCommitLearner": {
            "label": r"$(m)$ ExploreThenCommit bandit",
            "param_name": r"$m$",
            "param_grid": (grid := [2**k for k in range(0, 7)]),
            "learners": [ExploreThenCommitLearner(m) for m in grid],
            "plot_kwargs": {},
        },
        "EpsilonGreedyLearner": {
            "label": r"$(\epsilon)$ $\epsilon$-Greedy",
            "param_name": r"$\epsilon$",
            "param_grid": (grid := [2**k for k in range(-8, 0)]),
            "learners": [EpsilonGreedyLearner(eps) for eps in grid],
            "plot_kwargs": {},
        },
        "GreedyLearner": {
            "label": r"$(Q_0)$ Greedy with optimistic initialization, $\alpha=0.1$",
            "param_name": r"$Q_0$",
            "param_grid": (grid := [2**k for k in range(-3, 4)]),
            "learners": [EpsilonGreedyLearner(eps=0, alpha=0.1, init_reward=r) for r in grid],
            "plot_kwargs": {"ls": "--"},
        },
        "UCB1Learner": {
            "label": r"$(c)$ UCB1",
            "param_name": r"$c$",
            "param_grid": (grid := [2**k for k in range(-6, 1)]),
            "learners": [UCB1Learner(c) for c in grid],
            "plot_kwargs": {},
        },
        "GradientLearner": {
            "label": r"$(\alpha)$ Gradient bandit",
            "param_name": r"$\alpha$",
            "param_grid": (grid := [2**k for k in range(-4, 3)]),
            "learners": [GradientLearner(lr) for lr in grid],
            "plot_kwargs": {},
        },
        "ThompsonLearner": {
            "label": r"$(\alpha_0=\beta_0)$ Thompson bandit",
            "param_name": r"$\alpha_0=\beta_0$",
            "param_grid": (grid := [2**k for k in range(0, 8)]),
            "learners": [ThompsonLearner(a0=p, b0=p) for p in grid],
            "plot_kwargs": {},
        },
    }

    for config in tqdm(study.values()):
        mean_rewards = []
        for learner in config["learners"]:
            runs_rewards = evaluate_learner(learner, POTENTIAL_HITS, TIME_STEPS, TRIALS_PER_LEARNER)
            runs_rewards = 1 / TIME_STEPS * runs_rewards.sum(axis=1)
            mean_rewards.append(runs_rewards.mean())

        plt.plot(
            config["param_grid"],
            mean_rewards,
            color=learner.color,
            label=config["label"],
            **config["plot_kwargs"],
        )

    plt.legend(loc="best")
    plt.ylabel(f"Average reward after {TIME_STEPS} steps ({TRIALS_PER_LEARNER} trials)")
    plt.xlabel(", ".join(config["param_name"] for config in study.values()))
    plt.xscale("log", base=2)
    plt.show()


def task3():
    """
    Losowe wartości oczekiwane prawdopodobieństw (oraz tzw. regret)
    ---------------------------------------------------------------
    Badanie algorytmów dla tylko jednej instancji problemu utrudnia ich bardziej obiektywną ocenę.
    Rozszerz protokół testowy o generowanie za każdym razem nowego problemu (z osobnym układem
    prawdopodobieństw). Jak agregować wyniki uzyskane dla różnych problemów (w przypadku niektórych
    znacznie łatwiej o otrzymywanie nagród)? Opierając się o obserwację tego, na ile zgromadzone
    nagrody odbiegają od ich wartości oczekiwanej w sytuacji podejmowania za każdym razem optymalnej
    decyzji - jest to tzw. regret.
    """
    N_PROBLEMS = 500
    TIME_STEPS = 1_000
    TRIALS_PER_LEARNER = 1

    avg_regret: defaultdict[BanditLearner, NDArray] = defaultdict(lambda: 0)
    learners: list[BanditLearner] = [
        RandomLearner(),
        ExploreThenCommitLearner(m=50),
        EpsilonGreedyLearner(eps=0.1, init_reward=0),
        UCB1Learner(c=0.1),
        GradientLearner(lr=1.0),
        ThompsonLearner(a0=1, b0=1),
    ]

    potential_hits_list = [
        {
            "A": np.random.rand(),
            "B": np.random.rand(),
            "C": np.random.rand(),
            "D": np.random.rand(),
            "E": np.random.rand(),
        }
        for _ in range(N_PROBLEMS)
    ]

    for learner in tqdm(learners):
        avg_regret = compute_avg_regret(learner, potential_hits_list, TIME_STEPS, TRIALS_PER_LEARNER)
        plt.plot(avg_regret, c=learner.color, label=learner.name)

    plt.xlabel("Time")
    plt.ylabel(f"Cumulative avg. regret")
    plt.title(f"N_PROBLEMS={N_PROBLEMS} | TRIALS={TRIALS_PER_LEARNER}")
    plt.legend()
    plt.show()

    study = {
        "ExploreThenCommitLearner": {
            "label": r"$(m)$ ExploreThenCommit bandit",
            "param_name": r"$m$",
            "param_grid": (grid := [2**k for k in range(0, 7)]),
            "learners": [ExploreThenCommitLearner(m) for m in grid],
            "plot_kwargs": {},
        },
        "EpsilonGreedyLearner": {
            "label": r"$(\epsilon)$ $\epsilon$-Greedy",
            "param_name": r"$\epsilon$",
            "param_grid": (grid := [2**k for k in range(-8, 0)]),
            "learners": [EpsilonGreedyLearner(eps) for eps in grid],
            "plot_kwargs": {},
        },
        "GreedyLearner": {
            "label": r"$(Q_0)$ Greedy with optimistic initialization, $\alpha=0.1$",
            "param_name": r"$Q_0$",
            "param_grid": (grid := [2**k for k in range(-3, 4)]),
            "learners": [EpsilonGreedyLearner(eps=0, alpha=0.1, init_reward=r) for r in grid],
            "plot_kwargs": {"ls": "--"},
        },
        "UCB1Learner": {
            "label": r"$(c)$ UCB1",
            "param_name": r"$c$",
            "param_grid": (grid := [2**k for k in range(-6, 1)]),
            "learners": [UCB1Learner(c) for c in grid],
            "plot_kwargs": {},
        },
        "GradientLearner": {
            "label": r"$(\alpha)$ Gradient bandit",
            "param_name": r"$\alpha$",
            "param_grid": (grid := [2**k for k in range(-4, 3)]),
            "learners": [GradientLearner(lr) for lr in grid],
            "plot_kwargs": {},
        },
        "ThompsonLearner": {
            "label": r"$(\alpha_0=\beta_0)$ Thompson bandit",
            "param_name": r"$\alpha_0=\beta_0$",
            "param_grid": (grid := [2**k for k in range(0, 8)]),
            "learners": [ThompsonLearner(a0=p, b0=p) for p in grid],
            "plot_kwargs": {},
        },
    }

    for config in tqdm(study.values()):
        mean_regrets = []
        for learner in config["learners"]:
            runs_results = compute_avg_regret(learner, potential_hits_list, TIME_STEPS, TRIALS_PER_LEARNER)
            mean_regrets.append(runs_results[-1])

        plt.plot(
            config["param_grid"],
            mean_regrets,
            color=learner.color,
            label=config["label"],
            **config["plot_kwargs"],
        )

    plt.legend(loc="best")
    plt.ylabel(f"Average regret after {TIME_STEPS} steps")
    plt.xlabel(", ".join(config["param_name"] for config in study.values()))
    plt.xscale("log", base=2)
    plt.title(f"N_PROBLEMS={N_PROBLEMS} | TRIALS={TRIALS_PER_LEARNER}")
    plt.show()


def task4():
    """
    Bandyci niestacjonarni
    ----------------------
    Zmodyfikuj problem tak, by symulował stopniowy dryf preferencji odbiorców muzyki - niech z
    każdym krokiem czasowym prawdopodobieństwa przesłuchania utworu do końca zmieniają się o pewną
    niewielką wartość (np. losowaną z użyciem rozkładu normalnego) - w praktyce jest to pewne
    błądzenie losowe. Jak taka zmiana wpłynęła na wzajemną skuteczność badanych algorytmów? PS. W
    tym przypadku warto zmodyfikować algorytm zachłanny tak, by korzystał z pewnego learning rate -
    czyli zrealizować średnią wykładniczo ważoną aktualnością.
    """
    TIME_STEPS = 1_000
    TRIALS_PER_LEARNER = 50
    POTENTIAL_HITS = {
        "In Praise of Dreams": 0.8,
        "We Built This City": 0.9,
        "April Showers": 0.5,
        "Twenty Four Hours": 0.3,
        "Dirge for November": 0.1,
    }

    learners: list[BanditLearner] = [
        RandomLearner(),
        ExploreThenCommitLearner(m=16),
        EpsilonGreedyLearner(eps=2**-4, init_reward=0),
        UCB1Learner(c=2**-3),
        GradientLearner(lr=1.0),
        ThompsonLearner(a0=4, b0=4),
    ]
    potential_hits = [POTENTIAL_HITS]

    np.random.seed(0)
    for _ in range(TIME_STEPS):
        potential_hits.append(
            {k: np.clip(v + np.random.normal(scale=0.01), 0.0, 1.0) for k, v in potential_hits[-1].items()}
        )

    for arm in POTENTIAL_HITS:
        plt.plot([potential_hits[i][arm] for i in range(TIME_STEPS)], label=arm)
    plt.xlabel("Time")
    plt.ylabel("Thumb up probability")
    plt.legend()
    plt.show()

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots(nrows=2, ncols=3, sharey=True)

    fig1.tight_layout()
    fig2.tight_layout()

    for i, learner in enumerate(learners):
        runs_rewards = evaluate_learner(
            learner=learner,
            potential_hits=potential_hits,
            time_steps=TIME_STEPS,
            trials_per_learner=TRIALS_PER_LEARNER,
            bandit_class=DriftingTopHitBandit,
        )
        mean = runs_rewards.cumsum(axis=1).mean(axis=0)
        std = runs_rewards.cumsum(axis=1).std(axis=0)

        ax1.plot(mean, lw=0.8, c=learner.color, label=learner.name)
        ax2[i // 3, i % 3].fill_between(
            range(len(mean)),
            mean - std,
            mean + std,
            color=learner.color,
            label=learner.name,
            alpha=0.2,
        )
        ax2[i // 3, i % 3].legend(loc="upper left")
        ax2[i // 3, i % 3].set_xlim(0, TIME_STEPS)
        ax2[i // 3, i % 3].set_ylim(0, TIME_STEPS)
        ax2[i // 3, i % 3].set_xlabel("Time")
        ax2[i // 3, i % 3].grid(True)
        ax2[i // 3, i % 3].set_ylabel(f"Cumulative reward std.\n({TRIALS_PER_LEARNER} trials)")

    ax1.legend(loc="best")
    ax1.set_xlim(0, TIME_STEPS)
    ax1.set_ylim(0, TIME_STEPS)
    ax1.set_xlabel("Time")
    ax1.set_ylabel(f"Cumulative reward mean ({TRIALS_PER_LEARNER} trials)")

    plt.show()


if __name__ == "__main__":
    # task1()
    # task2()
    # task3()
    task4()
