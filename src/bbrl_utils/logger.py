import torch
from bbrl import instantiate_class


class Logger:
    def __init__(self, cfg):
        self.logger = instantiate_class(cfg)

    def add_log(self, log_string: float, loss: float, steps: int):
        self.logger.add_scalar(log_string, loss.item(), steps)

    def log_losses(
        self, critic_loss: float, entropy_loss: float, actor_loss: float, steps: int
    ):
        self.add_log("critic_loss", critic_loss, steps)
        self.add_log("entropy_loss", entropy_loss, steps)
        self.add_log("actor_loss", actor_loss, steps)

    def log_reward_losses(self, rewards: torch.Tensor, nb_steps):
        self.add_log("reward/mean", rewards.mean(), nb_steps)
        self.add_log("reward/max", rewards.max(), nb_steps)
        self.add_log("reward/min", rewards.min(), nb_steps)
        self.add_log("reward/median", rewards.median(), nb_steps)
