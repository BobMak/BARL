from dataclasses import dataclass, field
import torch as th
import gymnasium as gym

from BaseAgent import BaseAgent
from Architectures import str_to_arch


@dataclass
class SAC(BaseAgent):
    actor_arch:str="mlp"
    actor_arch_kwargs:dict=field(default_factory=dict)
    actor:th.nn.Module=field(init=False)
    critic_arch:str="mlp"
    critic_arch_kwargs:dict=field(default_factory=dict)
    critic:th.nn.Module=field(init=False)
    ent_coef:float=0.2
    def __post_init__(self):
        super().__post_init__()
        self.algo_name = 'SAC'
        self.log_hparams(self.get_all_kwargs())
        self.actor = str_to_arch[self.actor_arch](**self.actor_arch_kwargs)
        self.critic = str_to_arch[self.critic_arch](**self.critic_arch_kwargs)
        if self.env.action_space is gym.spaces.Discrete:
            self.exploration_policy = self._exploration_policy_disc
        else:
            self.exploration_policy = self._exploration_policy_cont

    def _evaluation_policy_disc(self, state):
        tstate = th.tensor(state, dtype=th.float32).to(self.device)
        action = th.argmax(self.actor(tstate))
        return action.cpu().detach().numpy()

    def _evaluation_policy_cont(self, state):
        tstate = th.tensor(state, dtype=th.float32).to(self.device)
        action, _ = self.actor(tstate)
        return action.cpu().detach().numpy()

    def _exploration_policy_disc(self, state):
        action_values = self.actor(th.tensor(state, dtype=th.float32).to(self.device))
        sampled_action = th.distributions.Categorical(action_values).sample()
        return sampled_action.cpu().detach().numpy()

    def _exploration_policy_cont(self, state):
        action_mu, action_vars = self.actor(th.tensor(state, dtype=th.float32).to(self.device))
        action = action_mu + th.sqrt(action_vars) * th.randn_like(action_mu)
        return action.cpu().detach().numpy()