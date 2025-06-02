import minari
from nanodt.agent import NanoDTAgent
from nanodt.utils import seed_libraries


def train_dt():
    seed_libraries(1234)
    dataset = minari.load_dataset("HalfCheetah-v4-expert-v0")

    agent = NanoDTAgent(device="cuda")
    agent.learn(dataset, reward_scale=1000.0)
    agent.save("output/dt/halfcheetah-expert-dt.pth")


if __name__ == "__main__":
    train_dt()
