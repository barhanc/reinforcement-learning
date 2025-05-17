import os

import torch
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from tqdm import trange
from ac import ActorCriticController


def train(controller: ActorCriticController, env: gym.Env, num_of_episodes: int, dirname: str):
    os.makedirs(f"plots/{dirname}/", exist_ok=True)
    os.makedirs(f"models/{dirname}/", exist_ok=True)

    past_errors = []
    past_rewards = []

    window_size = 50
    plot_frequency = 25

    for episode in trange(num_of_episodes):
        state, _ = env.reset()
        done = False
        reward_sum = 0.0
        errors_history = []

        while not done:
            action = controller.choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            controller.learn(state, reward, new_state, terminated)
            state = new_state
            reward_sum += reward
            errors_history.append(controller.last_error_squared)
            done = terminated or truncated

        past_rewards.append(reward_sum)
        past_errors.append(np.mean(errors_history))

        if episode % plot_frequency == 0 and len(past_rewards) >= window_size:
            fig, axs = plt.subplots(2)

            m = [np.mean(past_errors[i : i + window_size]) for i in range(len(past_errors) - window_size)]
            axs[0].plot(m, "tab:red")
            axs[0].set_ylabel("mean squared error")

            m = [np.mean(past_rewards[i : i + window_size]) for i in range(len(past_rewards) - window_size)]
            axs[1].plot(m, "tab:green")
            axs[1].set_ylabel("sum of rewards")

            fig.savefig(f"plots/{dirname}/learning_{episode}.png", bbox_inches="tight")
            fig.savefig(f"plots/{dirname}/learning.png", bbox_inches="tight")
            plt.close()

    env.close()

    # Save the model to disk
    torch.save(controller.actor.state_dict(), f"models/{dirname}/final_actor.model")
    torch.save(controller.critic.state_dict(), f"models/{dirname}/final_critic.model")


def evaluate(controller: ActorCriticController, env: gym.Env):
    state, _ = env.reset()
    done = False
    while not done:
        env.render()
        action = controller.choose_action(state)
        new_state, _, terminated, truncated, _ = env.step(action)
        state = new_state
        done = terminated or truncated


def main():
    print("\nCartPole-v1-backbone")
    env = gym.make("CartPole-v1", render_mode=None)
    controller = ActorCriticController(env, 0.00001, 0.99, {"h1": 1024, "h2": 256, "share_backbone": True})
    train(controller, env, num_of_episodes=2_000, dirname="CartPole-v1-backbone")

    print("\nCartPole-v1")
    env = gym.make("CartPole-v1", render_mode=None)
    controller = ActorCriticController(env, 0.00001, 0.99, {"h1": 1024, "h2": 256, "share_backbone": False})
    train(controller, env, num_of_episodes=2_000, dirname="CartPole-v1")

    print("\nCartPole-v1-small")
    env = gym.make("CartPole-v1", render_mode=None)
    controller = ActorCriticController(env, 0.00001, 0.99, {"h1": 128, "h2": 32, "share_backbone": False})
    train(controller, env, num_of_episodes=2_000, dirname="CartPole-v1-small")

    print("\nCartPole-v1-small-llr")
    env = gym.make("CartPole-v1", render_mode=None)
    controller = ActorCriticController(env, 0.01, 0.99, {"h1": 128, "h2": 32, "share_backbone": False})
    train(controller, env, num_of_episodes=2_000, dirname="CartPole-v1-small-llr")

    # print("LunarLander-v3")
    # env = gym.make("LunarLander-v3", render_mode=None)
    # controller = ActorCriticController(env, 5e-6, 0.99, {"h1": 1024, "h2": 256, "share_backbone": False})
    # train(controller, env, num_of_episodes=3_000, dirname="LunarLander-v3")

    # environment = gym.make("CartPole-v1", render_mode="human")
    # controller = ActorCriticController(environment, 1e-4, 0.99)
    # controller.actor.load_state_dict(torch.load("models/CartPole-v1-backbone/final_actor.model"))
    # controller.critic.load_state_dict(torch.load("models/CartPole-v1-backbone/final_critic.model"))
    # evaluate(controller, environment, num_of_episodes=10)


if __name__ == "__main__":
    main()
