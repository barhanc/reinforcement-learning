import torch
import torch.nn as nn
import gymnasium as gym

from torch import Tensor
from torch.distributions import Categorical

from typing import Optional
from numpy.typing import NDArray


class ActorCriticController:
    def __init__(self, environment: gym.Env, lr: float, discount_factor: float, params: dict):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.environment: gym.Env = environment
        self.discount_factor: float = discount_factor

        dim_state, *_ = environment.observation_space.shape
        dim_action = environment.action_space.n
        self.actor, self.critic = self.create_actor_critic_model(dim_state, dim_action, **params)

        params = set(self.actor.parameters()) | set(self.critic.parameters())
        self.optim = torch.optim.Adam(params, lr=lr)

        # Auxiliary variable, used for computing the target loss
        self.log_action_probability: Optional[Tensor] = None

        # Auxiliary variable, required for plotting
        self.last_error_squared: float = 0.0

    def create_actor_critic_model(
        self,
        d_in: int,
        d_out: int,
        h1: int,
        h2: int,
        share_backbone: bool,
    ) -> tuple[nn.Module, nn.Module]:
        # TODO: Create a neural network for actor and critic
        if share_backbone:
            backbone = nn.Sequential(
                nn.Linear(d_in, h1),
                nn.ReLU(),
                nn.LayerNorm(h1),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.LayerNorm(h2),
            )

            actor = nn.Sequential(
                backbone,
                nn.Linear(h2, d_out),
                nn.Softmax(1),
            ).to(self.device)

            critic = nn.Sequential(
                backbone,
                nn.Linear(h2, 1),
            ).to(self.device)

        else:
            actor = nn.Sequential(
                nn.Linear(d_in, h1),
                nn.ReLU(),
                nn.LayerNorm(h1),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.LayerNorm(h2),
                nn.Linear(h2, d_out),
                nn.Softmax(1),
            ).to(self.device)

            critic = nn.Sequential(
                nn.Linear(d_in, h1),
                nn.ReLU(),
                nn.LayerNorm(h1),
                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.LayerNorm(h2),
                nn.Linear(h2, 1),
            ).to(self.device)

        return actor, critic

    def choose_action(self, state: NDArray) -> int:
        state = torch.tensor(state).reshape(1, -1).float().to(self.device)

        # TODO: Choose action based on actor output
        m = Categorical(self.actor(state))
        action = m.sample()

        # TODO: Save action log probability
        self.log_action_probability = m.log_prob(action)

        return int(action.item())

    def learn(self, state: NDArray, reward: float, new_state: NDArray, terminal: bool) -> None:
        state: Tensor = torch.tensor(state).reshape(1, -1).float().to(self.device)
        new_state: Tensor = torch.tensor(new_state).reshape(1, -1).float().to(self.device)

        # TODO: Compute error for the critic model
        if not terminal:
            value = self.critic(new_state).item()
            error: Tensor = reward + self.discount_factor * value - self.critic(state)
        else:
            error: Tensor = reward - self.critic(state)

        loss = error**2 - error.item() * self.log_action_probability
        self.last_error_squared = error.item() ** 2

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
