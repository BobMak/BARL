import inspect
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
import gymnasium as gym
from typing import Optional, Union, Tuple, List
import tqdm
from typeguard import typechecked
from utils import auto_device, env_id_to_envs, find_torch_modules
from loggers.BaseLogger import BaseLogger
from Buffer import Buffer
# use get_type_hints to throw errors if the user passes in an invalid type:


@dataclass
@typechecked
class BaseAgent:
    env_id: str = None
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100_000
    gradient_steps: int = 1
    train_interval: int = 1
    max_grad_norm: float = 10
    learning_starts = 5_000
    device: Union[torch.device, str] = "auto"
    render: bool = False
    loggers: Tuple[callable] = ()
    logger_params: Tuple[dict] = ({},)
    log_interval: int = 1_000
    save_checkpoints: bool = False
    chkp_interval: int = 10_000
    chkp_latest_only: bool = True
    seed: Optional[int] = None
    eval_callbacks: Tuple[callable] = ()
    online: bool = False
    save_buffer: bool = False
    base_path: Optional[str] = './'
    env: Optional[gym.Env] = field(init=False)
    eval_env: Optional[gym.Env] = field(init=False)
    tot_env_steps: int = field(init=False)
    log_objs: List[BaseLogger] = field(init=False)
    buffer: Buffer = field(init=False)

    def __post_init__(self) -> None:
        # global number of training steps taken
        self.tot_env_steps = 0
        # current number of training steps taken within the learn()
        self.learn_env_steps = 0
        # current target of training steps to be taken within the learn()
        self.tot_learn_env_steps = 0
        is_atari = False
        permute_dims = False
        if isinstance(self.env_id, str):
            if 'ALE' in self.env_id or 'NoFrameskip' in self.env_id:
                is_atari=True
                permute_dims=True
        self.env, self.eval_env = env_id_to_envs(self.env_id, self.render, is_atari=is_atari, permute_dims=permute_dims)

        if hasattr(self.env.unwrapped.spec, 'id'):
            self.env_str = self.env.unwrapped.spec.id
        elif hasattr(self.env.unwrapped, 'id'):
            self.env_str = self.env.unwrapped.id
        else:
            self.env_str = str(self.env.unwrapped)

        self.log_objs = [lg(**lg_param) for lg, lg_param in zip(self.loggers, self.logger_params)]
        self.device = auto_device(self.device)
        self.chkp_interval = self.chkp_interval if self.save_checkpoints else -1

        self.eval_callbacks = [eb(self) for eb in self.eval_callbacks]
        self.avg_eval_rwd = None
        self.fps = None
        self.train_this_step = False
        self.num_episodes = 0
        self._n_updates = 0

        self.buffer = Buffer(
            buffer_size=self.buffer_size,
            state=self.env.observation_space.sample(),
            action=self.env.action_space.sample(),
            device=self.device
        )

    def log_hparams(self, hparam_dict):
        # Log the agent's hyperparameters:
        for logger in self.log_objs:
            logger.log_hparams(hparam_dict)

    def log_history(self, param, val, step):
        for logger in self.log_objs:
            logger.log_history(param, val, step)

    def _initialize_networks(self):
        raise NotImplementedError()

    def exploration_policy(self, state: np.ndarray):
        raise NotImplementedError()
    
    def evaluation_policy(self, state: np.ndarray):
        raise NotImplementedError()

    def calculate_loss(self, batch):
        raise NotImplementedError()

    def _train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        # Increase update counter
        self._n_updates += gradient_steps
        for _ in range(gradient_steps):
            # Sample a batch from the replay buffer:
            batch = self.buffer.sample(batch_size)

            loss = self.calculate_loss(batch)
            self.optimizer.zero_grad()

            # Clip gradient norm
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

    def learn(self, total_timesteps: int):
        """
        Train the agent for total_timesteps
        """
        # Start a timer to log fps:
        init_train_time = time.thread_time_ns()
        self.learn_env_steps = 0
        self.tot_learn_env_steps = total_timesteps
        with tqdm.tqdm(total=total_timesteps, desc="Training") as pbar:
            state, _ = self.env.reset()
            reward = 0.0
            while self.learn_env_steps < total_timesteps:
                action = self.exploration_policy(state)
                # Add the transition to the replay buffer:
                state = np.array([state])
                self.buffer.add(state, np.array([action]), reward, terminated)

                next_state, reward, terminated, truncated, infos = self.env.step(
                    action)

                next_state = np.array([next_state])
                self._on_step()
                ep_len += 1
                done = terminated or truncated
                self.rollout_reward += reward

                self.train_this_step = (self.train_interval == -1 and terminated) or \
                    (self.train_interval != -1 and self.learn_env_steps %
                    self.train_interval == 0)

                state = next_state
                if self.tot_env_steps % self.log_interval == 0:
                    train_time = (time.thread_time_ns() - init_train_time) / 1e9
                    train_fps = self.log_interval / train_time
                    self.log_history('time/train_fps', train_fps, self.learn_env_steps)
                    self.avg_eval_rwd = self.evaluate()
                    init_train_time = time.thread_time_ns()
                    pbar.update(self.log_interval)
                if self.save_checkpoints and self.tot_env_steps % self.chkp_interval == 0:
                    path = f"{str(self)}_{self.tot_env_steps}" if not self.chkp_latest_only else f"{str(self)}_latest"
                    self.save(os.path.join(self.base_path, path))

                if done:
                    state, _ = self.env.reset()
                    self.num_episodes += 1
                    self.rollout_reward = 0
                    ep_len = 0
                    self.log_history("rollout/ep_reward", self.rollout_reward, self.learn_env_steps)
                    self.log_history("rollout/episode_length", ep_len, self.learn_env_steps)

    def _on_step(self) -> None:
        """
        This method is called after every step in the environment
        """
        self.learn_env_steps += 1
        self.tot_env_steps += 1

        if self.train_this_step:
            if self.learn_env_steps > self.learning_starts:
                self._train(self.gradient_steps, self.batch_size)

    def evaluate(self, n_episodes=10) -> float:
        # run the current policy and return the average reward
        avg_reward = 0.
        n_steps = 0
        init_eval_time = time.process_time_ns()
        for ep in range(n_episodes):
            state, _ = self.eval_env.reset()
            done = False
            while not done:
                action = self.evaluation_policy(state)
                n_steps += 1
                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action)
                state = next_state
                avg_reward += reward
                done = terminated or truncated
                for callback in self.eval_callbacks:
                    callback(state=state, action=action, reward=reward, done=done, end=False)
        eval_time = (time.process_time_ns() - init_eval_time) / 1e9
        avg_reward /= n_episodes
        eval_fps = n_steps / eval_time
        self.eval_time = eval_time
        self.log_history('eval/avg_reward', avg_reward, self.learn_env_steps)
        self.log_history('eval/avg_episode_length', n_steps / n_episodes, self.learn_env_steps)
        self.log_history('eval/time', eval_time, self.learn_env_steps)
        self.log_history('eval/fps', eval_fps, self.learn_env_steps)
        for callback in self.eval_callbacks:
            callback(self, end=True)
        return avg_reward

    def get_all_kwargs(self):
        cls = self.__class__
        classes = deque([cls])
        kwargs = {}
        while classes:
            cls = classes.popleft()
            classes.extend(cls.__bases__)
            for str_arg in inspect.getfullargspec(cls.__init__).args[1:]:
                kwargs[str_arg] = getattr(self, str_arg)
            for str_arg in inspect.getfullargspec(cls.__init__).kwonlyargs:
                kwargs[str_arg] = getattr(self, str_arg)
        return kwargs

    def save(self, path=None):
        if path is None:
            path = os.path.join(self.base_path, str(self))
        # save the number of time steps:
        kwargs = self.get_all_kwargs()
        total_state = {
            "kwargs": kwargs,
            "state_dicts": find_torch_modules(self),
            "class": self.__class__.__name__,
            "tot_env_steps": self.tot_env_steps,
            "continue_training": True
        }
        if self.save_buffer:
            total_state['buffer'] = self.buffer
        # if the path is a directory, make :
        if '/' in path:
            bp = path.split('/')
            base_path = os.path.join(*bp[:-1])
            if not os.path.exists(base_path):
                os.makedirs(base_path)
        torch.save(total_state, path)

    @staticmethod
    def load(path, **new_kwargs):
        state = torch.load(path)
        cls = BaseAgent
        for cls_ in BaseAgent.__subclasses__():
            if cls_.__name__ == state['class']:
                cls = cls_
        args = state['kwargs'].get('args', ())
        kwargs = state['kwargs']
        kwargs.update(new_kwargs)
        agent = cls(*args, **kwargs)
        for k, v in state['state_dicts'].items():
            attrs = k.split('.')
            module = agent
            for attr in attrs:
                module = getattr(module, attr)
            module.load_state_dict(v)
        if 'buffer' in state:
            agent.buffer = state['buffer']
        agent.tot_env_steps = state['tot_env_steps']
        agent.continue_training = state['continue_training']
        return agent

    def __str__(self):
        return f"{self.__class__.__name__}_{self.env_str}"