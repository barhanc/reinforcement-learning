Autor: Bartosz Hanc

# Decision Transformer - Raport

## 🔧 Etap 1: Trening eksperta

* Środowisko: HalfCheetah-v4 (MuJoCo)

* Algorytm: SAC (Stable Baselines3)

* Czas treningu: 1 milion kroków

* Oprogramowanie:
  - stable-baselines3
  - gymnasium
  - RL Baselines3 Zoo

```bash
python3 -m rl_zoo3.train -conf hyperparameters/sac.yml --env HalfCheetah-v4 --algo sac -P
```

## 🎥 Etap 2: Zbieranie demonstracji

* Do zapisu trajektorii wykorzystano skrypt `examples/collect_dataset.py`
```bash
python3 collect_dataset.py --env-name HalfCheetah-v4 \
  --expert-path logs/sac/HalfCheetah-v4_1/HalfCheetah-v4.zip \
  --n-timesteps 1_000_000
```

## 🤖 Etap 3: Trening Decision Transformera

* Biblioteka: nanoDT

* Zbiór danych: minari/HalfCheetah-v4-expert-v0

* Czas treningu: ~1h

* NOTE: Należy wcześniej utworzyć strukturę katalogów `output/dt/` (inaczej czeka nas przykra
  niespodzianka po długim oczekiwaniu na zakończenie treningu)

```python
import minari
from nanodt.agent import NanoDTAgent
from nanodt.utils import seed_libraries

def train_dt():
    seed_libraries(1234)
    dataset = minari.load_dataset("HalfCheetah-v4-expert-v0")

    agent = NanoDTAgent(device="cuda")
    agent.learn(dataset, reward_scale=1000.0)
    agent.save("output/dt/halfcheetah-expert-dt.pth")
```

## 📊 Etap 4: Ewaluacja

```python
import gymnasium as gym
from stable_baselines3 import SAC
from nanodt.agent import NanoDTAgent

def evaluate_dt(n_episodes=5):
    env = gym.make("HalfCheetah-v4", render_mode="human")
    agent = NanoDTAgent.load("output/dt/halfcheetah-expert-dt.pth")

    for i in range(n_episodes):
        agent.reset(target_return=6000)
        obs, _ = env.reset()
        done = False
        accumulated_rew = 0
        
        while not done:
            action = agent.act(obs)
            obs, rew, ter, tru, _ = env.step(action)
            done = ter or tru
            accumulated_rew += rew

        print(f"[Episode {i}] Accumulated rew: ", accumulated_rew)

    env.close()


def evaluate_expert(n_episodes=5):
    env = gym.make("HalfCheetah-v4", render_mode="human")
    model = SAC.load("logs/sac/HalfCheetah-v4_1/HalfCheetah-v4.zip")

    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        accumulated_rew = 0

        while not done:
            action, _ = model.predict(obs)
            obs, rew, ter, tru, _ = env.step(action)
            done = ter or tru
            accumulated_rew += rew

        print(f"[Episode {i}] Accumulated rew: ", accumulated_rew)

    env.close()
```

| Agent                | Średnia nagroda | Komentarz                                     |
| -------------------- | --------------- | --------------------------------------------- |
| Ekspert (SAC)        | \~6500          | Stabilne wyniki                               |
| Decision Transformer | \~4600          | Uczy się dobrze, ale słabiej niż ekspert      |

* DT nie zawsze osiąga maksymalne nagrody, ale zbliża się do eksperta.