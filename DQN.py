from dataclasses import dataclass, field
from typing import Optional, Union
import gymnasium as gym
import numpy as np
import torch
from Architectures import make_atari_nature_cnn, make_mlp, str_to_arch
from BaseAgent import BaseAgent
from utils import polyak


@dataclass
class DQN(BaseAgent):
    architecture: Union[str, torch.nn.Module, callable] = "mlp"
    gamma: float = 0.99
    minimum_epsilon: float = 0.05
    exploration_fraction: float = 0.5
    initial_epsilon: float = 1.0
    use_target_network: bool = False
    target_update_interval: Optional[int] = None
    polyak_tau: Optional[float] = None
    architecture_kwargs: dict = field(default_factory=dict)
    nA: int = 0
    optimizer: torch.optim.Optimizer = field(init=False)
    def __post_init__(self):
        super().__post_init__()
        self.epsilon = self.initial_epsilon
        self.algo_name = 'SQL'
        self.nA = self.env.action_space.n
        if type(self.architecture) == str:
            arch = str_to_arch[self.architecture]
        else:
            arch = self.architecture
        self.architecture_kwargs['input_dim'] = (
            self.env.observation_space.shape)[0] if arch == make_mlp else self.env.observation_space.shape
        self.architecture_kwargs['output_dim'] = (
            self.env.action_space.n) if isinstance(self.env.action_space, gym.spaces.Discrete) \
            else self.env.action_space.shape
        self.log_hparams(self.get_all_kwargs())
        self.online_qs = arch(**self.architecture_kwargs)
        self.model = self.online_qs

        if self.use_target_network:
            # Make another instance of the architecture for the target network:
            self.target_qs = arch(**self.architecture_kwargs)
            self.target_qs.load_state_dict(self.online_qs.state_dict())
            if self.polyak_tau is not None:
                assert 0 <= self.polyak_tau <= 1, "Polyak tau must be in the range [0, 1]."
            else:
                print("WARNING: No polyak tau specified for soft target updates. Using default tau=1 for hard updates.")
                self.polyak_tau = 1.0

            if self.target_update_interval is None:
                print("WARNING: Target network update interval not specified. Using default interval of 1 step.")
                self.target_update_interval = 1
        # Alias the "target" with online net if target is not used:
        else:
            self.target_qs = self.online_qs
            # Raise a warning if update interval is specified:
            if self.target_update_interval is not None:
                print("WARNING: Target network update interval specified but target network is not used.")

        # Make (all) qs learnable:
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _on_step(self) -> None:
        super()._on_step()

        # Update epsilon:
        self.epsilon = max(self.minimum_epsilon, (self.initial_epsilon - self.learn_env_steps / self.tot_learn_env_steps / self.exploration_fraction))

        if self.learn_env_steps % self.log_interval == 0:
            self.log_history("train/epsilon", self.epsilon, self.learn_env_steps)

        # Periodically update the target network:
        if self.use_target_network and self.learn_env_steps % self.target_update_interval == 0:
            # Use Polyak averaging as specified:
            polyak(self.online_qs, self.target_qs, self.polyak_tau)


    def exploration_policy(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.evaluation_policy(state)
    

    def evaluation_policy(self, state: np.ndarray) -> int:
        # Get the greedy action from the q values:
        qvals = self.online_qs(state)
        qvals = qvals.squeeze()
        return torch.argmax(qvals).item()
    

    def calculate_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        actions = actions.long()
        dones = dones.float()
        curr_q = self.online_qs(states).squeeze().gather(1, actions.long())
        with torch.no_grad():
            if isinstance(self.env.observation_space, gym.spaces.Discrete):
                states = states.squeeze()
                next_states = next_states.squeeze()

            next_qs = self.target_qs(next_states)
            
            next_v = torch.max(next_qs, dim=-1).values
            next_v = next_v.reshape(-1, 1)

            # Backup equation:
            expected_curr_q = rewards + self.gamma * next_v * (1-dones)

        # Calculate the q ("critic") loss:
        loss = 0.5*torch.nn.functional.mse_loss(curr_q, expected_curr_q)

        if self.learn_env_steps % self.log_interval == 0:
            self.log_history("train/online_q_mean", curr_q.mean().item(), self.learn_env_steps)
            self.log_history("train/loss", loss.item(), self.learn_env_steps)

        return loss

    def __str__(self):
        return f"{self.__class__.__name__}_{self.env_str}_g{self.gamma}_eps{self.initial_epsilon}_bs{self.batch_size}"


if __name__ == '__main__':
    from loggers.TensorboardLogger import TensorboardLogger
    # from loggers.StdLogger import StdLogger
    # logger = StdLogger
    # logger_params = {}
    logger = TensorboardLogger
    logger_params = {"log_dir":'logs/atari'}
    env = 'CartPole-v1'
    agent = DQN(env, 
                architecture=make_mlp,
                architecture_kwargs={'input_dim': gym.make(env).observation_space.shape[0],
                                     'output_dim': gym.make(env).action_space.n,
                                     'hidden_dims': [256, 256]},
                loggers=(logger,),
                logger_params=(logger_params,),
                learning_rate=0.001,
                train_interval=1,
                gradient_steps=4,
                batch_size=64,
                use_target_network=True,
                target_update_interval=10,
                polyak_tau=1.0,
                learning_starts=1000,
                log_interval=500,
                )
    # agent.learn(total_timesteps=50_000)
    # agent.save("saved_DQN")
    saved_agent = DQN.load("saved_DQN")
    saved_agent.evaluate(5)
    # visualize evaluation
    env = gym.make(env, render_mode='human')
    for _ in range(3):
        state, _ = env.reset()
        done = False
        while not done:
            action = saved_agent.evaluation_policy(state)
            state, _, done, _, _ = env.step(action)
            env.render()

