import copy
from abc import ABC
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from omegaconf import OmegaConf
import torch
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.agents import Agent, Agents, TemporalAgent
from bbrl.agents.gymnasium import ParallelGymAgent, make_env, record_video
from bbrl.workspace import Workspace

from bbrl_utils.logger import Logger
from bbrl_utils.notebook import tqdm, video_display, outputs_dir


class RLBase(ABC):
    """Base class for Reinforcement learning algorithms

    This class deals with common processing:

    - defines the logger, the train and evaluation agents
    - defines how to evaluate a policy

    This class:

    1. Initializes the environment (random seed, logger, evaluation environment)
    2. Defines a `evaluate` method which evaluates the current `eval_policy` and
    keeps the best agent so far 3. Defines a `visualize_best` method which
    displays the behavior of the best agent

    **Subclasses need to define** `self.train_policy` (and optionally
    `self.eval_policy`), two BBRL agents which respectively choose actions when
    training and evaluating.

    The behavior of `RLBase` is controlled by the following configuration
    variables:
    - `base_dir` defines the directory subpath used when outputting losses
    during training as well as other outputs (serialized agent, global statistics,
    etc.)
    - `algorithm.seed` defines the random seed used (to initialize the
    agent and the environment)
    - `gym_env` defines the gymnasium environment,
    and in particular `gym_env.env_name` the name of the gymnasium environment
    - `logger` defines what type of logger is used to log the different values
    associated with learning
    - `algorithm.eval_interval` defines the number of
    observed transitions between each evaluation of the agent
    """

    #: The configuration
    cfg: Any

    #: The evaluation environment deals with the last action, and produces a new
    # state of the environment
    eval_env: Agent

    #: The training policy
    train_policy: Agent

    #: The evaluation policy (if not defined, uses the training policy)
    eval_policy: Agent

    def __init__(self, cfg, env_wrappers=[]):
        # Basic initialization
        self.cfg = cfg
        self.make_env = partial(make_env, cfg.gym_env.env_name, wrappers=env_wrappers)
        torch.manual_seed(cfg.algorithm.seed)

        # Sets the base directory and logger directory
        base_dir = Path(self.cfg.base_dir)
        self.base_dir = outputs_dir() / Path(self.cfg.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize the logger class
        if not (logger_cfg := cfg.get("logger", None)):
            logger_cfg = OmegaConf.create(
                {
                    "classname": "bbrl.utils.logger.TFLogger",
                    "cache_size": 10000,
                    "every_n_seconds": 10,
                    "verbose": False,
                }
            )

        if not hasattr(logger_cfg, "log_dir"):
            logger_cfg.log_dir = str(outputs_dir() / "tblogs" / base_dir)

        self.logger = Logger(logger_cfg)

        # Subclasses have to define the training and eval policies
        self.train_policy = None
        self.eval_policy = None

        # Sets up the evaluation environment
        self.eval_env = ParallelGymAgent(self.make_env, cfg.algorithm.nb_evals).seed(
            cfg.algorithm.seed
        )

        # Initialize values
        self.last_eval_step = 0
        self.nb_steps = 0
        self.best_policy = None
        self.best_reward = -torch.inf
        self.running_reward = -torch.inf
        self.running_reward_alpha = 0.99

        # Records the rewards
        self.eval_rewards = []

    @cached_property
    def train_agent(self):
        """Returns the training agent

        The agent is composed of a policy agent and the training environment.
        This method assumes that `self.train_policy` has been setup
        """
        assert (
            self.train_policy is not None
        ), "The train_policy property is not defined before the policy is set"
        return TemporalAgent(Agents(self.train_env, self.train_policy))

    @cached_property
    def eval_agent(self):
        """Returns the evaluation agent

        The agent is composed of a policy agent and the evaluation environment

        Uses `self.eval_policy` (or `self.train_policy` if not defined)

        """
        assert (
            self.eval_policy is not None or self.train_policy is not None
        ), "eval_agent property is not defined before the policy is set"
        return TemporalAgent(
            Agents(
                self.eval_env,
                self.eval_policy if self.eval_policy is not None else self.train_policy,
            )
        )

    def evaluate(self, force=False):
        """Evaluate the current policy `self.eval_policy`

        Evaluation is conducted every `cfg.algorithm.eval_interval` steps, and
        we keep a copy of the best agent so far in `self.best_policy`

        Returns True if the current policy is the best so far
        """
        if force or (
            (self.nb_steps - self.last_eval_step) > self.cfg.algorithm.eval_interval
        ):
            self.last_eval_step = self.nb_steps
            eval_workspace = Workspace()
            self.eval_agent(eval_workspace, t=0, stop_variable="env/done")
            rewards = eval_workspace["env/cumulated_reward"][-1]
            self.register_evaluation(rewards)

    def register_evaluation(self, rewards: torch.Tensor, policy=None):
        """Directly registers an evaluation

        :param rewards: The rewards obtained
        :param policy: The policy agent
        """
        self.logger.log_reward_losses(rewards, self.nb_steps)

        if getattr(self.cfg, "collect_stats", False):
            self.eval_rewards.append(rewards)

        rewards_mean = rewards.mean()

        if self.running_reward == -torch.inf:
            self.running_reward = rewards_mean
        else:
            self.running_reward = (
                1.0 - self.running_reward_alpha
            ) * rewards_mean + self.running_reward_alpha * self.running_reward

        if rewards_mean > self.best_reward:
            self.best_policy = copy.deepcopy(
                self.eval_policy if policy is None else policy
            )
            self.best_reward = rewards_mean
            return True

    def save_stats(self):
        """Save reward statistics into `stats.npy`"""
        if getattr(self.cfg, "collect_stats", False) and self.eval_rewards:
            data = torch.stack(self.eval_rewards, axis=-1)
            with (self.base_dir / "stats.npy").open("wt") as fp:
                np.savetxt(fp, data.numpy())

    def visualize_best(self):
        """Visualize the best agent"""
        env = self.make_env(render_mode="rgb_array")
        path = self.base_dir / "best_agent.mp4"
        print(f"Video of best agent recorded in {path}")
        record_video(env, self.best_policy, path)
        return video_display(str(path.absolute()))


class EpochBasedAlgo(RLBase):
    """RL environment when using transition buffers"""

    train_agent: TemporalAgent

    """Base class for RL experiments with full episodes"""

    def __init__(self, cfg, env_wrappers=[]):
        """Creates a new epoch-based RL algorithm

        :param cfg: The configuration
        :param env_wrappers: A list of factories, defaults to []
        """
        super().__init__(cfg, env_wrappers=env_wrappers)

        # We use a non-autoreset workspace
        self.train_env = ParallelGymAgent(
            partial(self.make_env, autoreset=True),
            cfg.algorithm.n_envs,
        ).seed(cfg.algorithm.seed)

        # Configure the workspace to the right dimension
        # Note that no parameter is needed to create the workspace.
        self.replay_buffer = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    def iter_replay_buffers(self):
        """Loop over transition buffers

        `iter_replay_buffers` provides an easy access to the replay buffer when
        learning. Its behavior depends on several configuration values:

        - `cfg.algorithm.max_epochs` defines the number of times the agent is used to
        collect transitions
        - `cfg.algorithm.learning_starts` defines the number of transitions before
        learning starts

        Using `iter_replay_buffers` is simple:

        ```py
        class MyAlgo(EpochBasedAlgo):
            def __init__(self, cfg):
                super().__init__(cfg)

                # Define the train and evaluation policies
                # (the agents compute the workspace `action` variable)
                self.train_policy = ...
                self.eval_policy = ...

        rl_algo = MyAlgo(cfg)
        for rb in iter_replay_buffers(rl_algo):
            # rb is a workspace containing transitions
            ...
        ```
        """
        train_workspace = Workspace()

        epochs_pb = tqdm(range(self.cfg.algorithm.max_epochs))
        for epoch in epochs_pb:
            # This is the tricky part with transition buffers. The difficulty lies in the
            # copy of the last step and the way to deal with the n_steps return.
            #
            # The call to `train_agent(workspace, t=1, n_steps=cfg.algorithm.n_timesteps -
            # 1, stochastic=True)` makes the agent run a number of steps in the workspace.
            # In practice, it calls the
            # [`__call__(...)`](https://github.com/osigaud/bbrl/blob/master/src/bbrl/agents/agent.py#L59)
            # function which makes a forward pass of the agent network using the workspace
            # data and updates the workspace accordingly.
            #
            # Now, if we start at the first epoch (`epoch=0`), we start from the first step
            # (`t=0`). But when subsequently we perform the next epochs (`epoch>0`), we must
            # not forget to cover the transition at the border between the previous epoch
            # and the current epoch. To avoid this risk, we copy the information from the
            # last time step of the previous epoch into the first time step of the next
            # epoch. This is explained in more details in [a previous
            # notebook](https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5).
            if epoch == 0:
                # First run: we start from scratch
                self.train_agent(
                    train_workspace,
                    t=0,
                    n_steps=self.cfg.algorithm.n_steps + 1,
                    stochastic=True,
                )
            else:
                # Other runs: we copy the last step and start from there
                train_workspace.zero_grad()
                train_workspace.copy_n_last_steps(1)
                self.train_agent(
                    train_workspace,
                    t=1,
                    n_steps=self.cfg.algorithm.n_steps,
                    stochastic=True,
                )

            self.nb_steps += self.cfg.algorithm.n_steps * self.cfg.algorithm.n_envs

            # Add transitions to buffer
            transition_workspace = train_workspace.get_transitions()
            self.replay_buffer.put(transition_workspace)
            if self.replay_buffer.size() > self.cfg.algorithm.learning_starts:
                yield self.replay_buffer

            # Eval
            epochs_pb.set_description(
                f"nb_steps: {self.nb_steps}, "
                f"best reward: {self.best_reward: .2f}, "
                f"running reward: {self.running_reward: .2f}"
            )


class EpisodicAlgo(RLBase):
    """Base class for RL experiments with full episodes"""

    def __init__(self, cfg, autoreset=False, env_wrappers=[]):
        super().__init__(cfg, env_wrappers=env_wrappers)

        self.train_env = ParallelGymAgent(
            partial(
                self.make_env,
                autoreset=autoreset,
            ),
            cfg.algorithm.n_envs,
        ).seed(cfg.algorithm.seed)

    def iter_episodes(self) -> Iterator[Workspace]:
        """Iterate over episodes without auto-reset.

        When using more than one environment, some workspaces will contain
        duplicated entries at the end with `env/done` set to true.
        """
        pbar = tqdm(range(self.cfg.algorithm.max_epochs))

        for self.epoch in pbar:
            # Collect samples
            train_workspace = Workspace()
            self.train_agent(train_workspace, t=0, stop_variable="env/done")

            # Update the number of steps
            self.nb_steps += int((~train_workspace["env/done"]).sum())

            # Perform a learning step
            yield train_workspace

            # Eval
            pbar.set_description(
                f"nb_steps: {self.nb_steps}, best reward: {self.best_reward: .2f}"
            )

    def iter_partial_episodes(self, episode_steps: int = None) -> Iterator[Workspace]:
        """Iterate over partial episodes

        Returns an iterator over workspaces. Each workspace contains exactly
        `episode_steps` steps (uses `algorithm.n_steps` if not specified)
        """
        episode_steps = episode_steps or self.cfg.algorithm.n_steps
        pbar = tqdm(range(self.cfg.algorithm.max_epochs))
        train_workspace = Workspace()

        for self.epoch in pbar:
            if self.epoch > 0:
                train_workspace.zero_grad()
                train_workspace.copy_n_last_steps(1)
                self.train_agent(train_workspace, t=1, n_steps=episode_steps - 1)
            else:
                self.train_agent(train_workspace, t=0, n_steps=episode_steps)

            self.nb_steps += int((~train_workspace["env/done"]).sum())
            yield train_workspace

            pbar.set_description(
                f"nb_steps: {self.nb_steps}, "
                f"reward: best={self.best_reward: .2f}, "
                f"running={self.running_reward: .2f}"
            )


iter_episodes = EpisodicAlgo.iter_episodes
iter_replay_buffers = EpochBasedAlgo.iter_replay_buffers
iter_partial_episodes = EpisodicAlgo.iter_partial_episodes
