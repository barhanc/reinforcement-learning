import gymnasium as gym

from stable_baselines3 import SAC
from nanodt.agent import NanoDTAgent


def evaluate_dt():
    env = gym.make("HalfCheetah-v4", render_mode="human")
    agent = NanoDTAgent.load("output/dt/halfcheetah-expert-dt.pth")

    for _ in range(5):
        agent.reset(target_return=6000)
        obs, info = env.reset()
        done = False
        accumulated_rew = 0
        while not done:
            action = agent.act(obs)
            obs, rew, ter, tru, info = env.step(action)
            done = ter or tru
            accumulated_rew += rew

        print("Accumulated reward: ", accumulated_rew)

    env.close()


def evaluate_expert():
    env = gym.make("HalfCheetah-v4", render_mode="human")
    model = SAC.load("logs/sac/HalfCheetah-v4_1/HalfCheetah-v4.zip")

    for i in range(3):
        obs, info = env.reset()
        done = False
        accumulated_rew = 0

        while not done:
            action, _ = model.predict(obs)
            obs, rew, ter, tru, info = env.step(action)
            done = ter or tru
            accumulated_rew += rew

        print("Accumulated rew: ", accumulated_rew)

    env.close()


if __name__ == "__main__":
    print("[Expert]")
    evaluate_expert()

    # print("[Decision Transformer]")
    # evaluate_dt()
