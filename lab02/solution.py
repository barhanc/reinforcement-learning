from __future__ import annotations

import random
import collections

import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as skl_preprocessing

from itertools import product
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from problem import Action, available_actions, Corner, Driver, Experiment, Environment, State

MAX_LEARNING_STEPS = 500
ALMOST_INFINITE_STEP = 100000


class RandomDriver(Driver):
    def __init__(self):
        self.current_step: int = 0

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        return random.choice(available_actions(state))

    def control(self, state: State, last_reward: int) -> Action:
        self.current_step += 1
        return random.choice(available_actions(state))

    def finished_learning(self) -> bool:
        return self.current_step > MAX_LEARNING_STEPS


class OffPolicyNStepSarsaDriver(Driver):
    def __init__(self, step_size: float, step_no: int, experiment_rate: float, discount_factor: float) -> None:
        self.step_size: float = step_size
        self.step_no: int = step_no
        self.experiment_rate: float = experiment_rate
        self.discount_factor: float = discount_factor
        self.q: dict[tuple[State, Action], float] = collections.defaultdict(float)
        self.current_step: int = 0
        self.final_step: int = ALMOST_INFINITE_STEP
        self.finished: bool = False
        self.states: dict[int, State] = dict()
        self.actions: dict[int, Action] = dict()
        self.rewards: dict[int, int] = dict()
        self.eval: bool = False

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        self.states[self._access_index(self.current_step)] = state
        action = self._select_action(self.epsilon_greedy_policy(state, available_actions(state)))
        self.actions[self._access_index(self.current_step)] = action
        self.final_step = ALMOST_INFINITE_STEP
        self.finished = False
        return action

    def control(self, state: State, last_reward: int) -> Action:
        if self.eval:
            self.current_step += 1
            self.finished = self.current_step > MAX_LEARNING_STEPS
            return self._select_action(self.greedy_policy(state, available_actions(state)))

        if self.current_step < self.final_step:
            self.rewards[self._access_index(self.current_step + 1)] = last_reward
            self.states[self._access_index(self.current_step + 1)] = state
            if self.final_step == ALMOST_INFINITE_STEP and (
                last_reward == 0 or self.current_step == MAX_LEARNING_STEPS
            ):
                self.final_step = self.current_step
            action = self._select_action(self.epsilon_greedy_policy(state, available_actions(state)))
            self.actions[self._access_index(self.current_step + 1)] = action
        else:
            action = Action(0, 0)

        update_step = self.current_step - self.step_no + 1
        if update_step >= 0:
            return_value_weight = self._return_value_weight(update_step)
            return_value = self._return_value(update_step)
            state_t = self.states[self._access_index(update_step)]
            action_t = self.actions[self._access_index(update_step)]

            # TODO: Tutaj trzeba zaktualizować tablicę wartościującą akcje Q
            if update_step + self.step_no < self.final_step:
                return_value += self.discount_factor**self.step_no * self.q[state, action]

            delta = return_value_weight * (return_value - self.q[state_t, action_t])
            self.q[state_t, action_t] += self.step_size * delta

        if update_step == self.final_step - 1:
            self.finished = True

        self.current_step += 1
        return action

    def _return_value(self, update_step: int) -> float:
        # TODO: Tutaj trzeba policzyć zwrot G
        return_value = 0.0
        t_min = update_step + 1
        t_max = min(update_step + self.step_no + 1, self.final_step + 1)

        for i in range(t_min, t_max):
            reward_i = self.rewards[self._access_index(i)]
            return_value += self.discount_factor ** (i - update_step - 1) * reward_i

        return return_value

    def _return_value_weight(self, update_step: int) -> float:
        # TODO: Tutaj trzeba policzyć korektę na różne prawdopodobieństwa p
        # (ponieważ uczymy poza-polityką)
        return_value_weight = 1.0
        t_min = update_step + 1
        t_max = min(update_step + self.step_no, self.final_step)

        for i in range(t_min, t_max):
            state_i = self.states[self._access_index(i)]
            action_i = self.actions[self._access_index(i)]

            p = self.greedy_policy(state_i, available_actions(state_i))
            b = self.epsilon_greedy_policy(state_i, available_actions(state_i))

            return_value_weight *= p[action_i] / b[action_i]

        return return_value_weight

    def finished_learning(self) -> bool:
        return self.finished

    def _access_index(self, index: int) -> int:
        return index % (self.step_no + 1)

    @staticmethod
    def _select_action(actions_distribution: dict[Action, float]) -> Action:
        actions = list(actions_distribution.keys())
        probabilities = list(actions_distribution.values())
        i = np.random.choice(list(range(len(actions))), p=probabilities)
        return actions[i]

    def epsilon_greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        # TODO: tutaj trzeba ustalic prawdopodobieństwa wyboru akcji według polityki ϵ-zachłannej
        eps = self.experiment_rate
        p_random = self._random_probabilities(actions)
        p_greedy = self._greedy_probabilities(state, actions)
        probas = eps * p_random + (1 - eps) * p_greedy
        return {action: proba for action, proba in zip(actions, probas)}

    def greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        probabilities = self._greedy_probabilities(state, actions)
        return {action: probability for action, probability in zip(actions, probabilities)}

    def _greedy_probabilities(self, state: State, actions: list[Action]) -> np.ndarray:
        values = [self.q[state, action] for action in actions]
        maximal_spots = (values == np.max(values)).astype(float)
        return self._normalise(maximal_spots)

    @staticmethod
    def _random_probabilities(actions: list[Action]) -> np.ndarray:
        maximal_spots = np.array([1.0 for _ in actions])
        return OffPolicyNStepSarsaDriver._normalise(maximal_spots)

    @staticmethod
    def _normalise(probabilities: np.ndarray) -> np.ndarray:
        return skl_preprocessing.normalize(probabilities.reshape(1, -1), norm="l1")[0]


def worker(n: int, α: float) -> tuple[int, float, float]:
    experiment = Experiment(
        environment=Environment(
            corner=Corner(name="corner_c"),
            steering_fail_chance=0.01,
        ),
        driver=OffPolicyNStepSarsaDriver(
            step_no=n,
            step_size=α,
            experiment_rate=0.05,
            discount_factor=1.00,
        ),
        number_of_episodes=10,
        verbose=False,
    )
    t = experiment.run()
    return n, α, t


def main():
    # CORNER B
    # =====================================================
    # experiment = Experiment(
    #     environment=Environment(
    #         corner=Corner(name="corner_b"),
    #         steering_fail_chance=0.01,
    #     ),
    #     driver=OffPolicyNStepSarsaDriver(
    #         step_no=5,
    #         step_size=0.3,
    #         experiment_rate=0.05,
    #         discount_factor=1.00,
    #     ),
    #     number_of_episodes=4_000,
    #     savedir="plots_corner_b",
    # )
    # print("CORNER B")
    # time = experiment.run()
    # print(f"Average time = {time:.2f}\n")

    # CORNER C
    # =====================================================
    # experiment = Experiment(
    #     environment=Environment(
    #         corner=Corner(name="corner_c"),
    #         steering_fail_chance=0.01,
    #     ),
    #     driver=OffPolicyNStepSarsaDriver(
    #         step_no=5,
    #         step_size=0.3,
    #         experiment_rate=0.05,
    #         discount_factor=1.00,
    #     ),
    #     number_of_episodes=4_000,
    #     savedir="plots_corner_c",
    # )
    # print("CORNER C")
    # time = experiment.run()
    # print(f"Average time = {time:.2f}\n")

    # PARAMETRIC STUDY
    # =====================================================
    ns = [2, 4, 8, 16]
    αs = np.arange(0, 1.1, 0.1)
    results = defaultdict(lambda: None)

    with ProcessPoolExecutor() as executor:
        for n, α, t in tqdm(executor.map(worker, *zip(*product(ns, αs))), total=len(ns) * len(αs)):
            results[n, α] = t

    # --- Plot results
    for n in ns:
        plt.plot(αs, [results[n, α] for α in αs], label=n)
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Average time")
    plt.legend()
    plt.savefig("parametric_study.png", dpi=500)

    # CORNER D
    # =====================================================
    experiment = Experiment(
        environment=Environment(
            corner=Corner(name="corner_d"),
            steering_fail_chance=0.01,
        ),
        driver=OffPolicyNStepSarsaDriver(
            step_no=5,
            step_size=0.8,
            experiment_rate=0.05,
            discount_factor=1.00,
        ),
        number_of_episodes=10_000,
        savedir="plots_corner_d",
    )
    print("CORNER D")
    time = experiment.run()
    print(f"Average time = {time:.2f}\n")


if __name__ == "__main__":
    main()
