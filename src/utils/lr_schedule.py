import torch
import math
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR



def linear_warmup_cosine_annealingLR(optimizer: torch.optim.Optimizer, max_steps:int, linear_warmup_rate:float=0.05, min_lr:float=5e-4):
    assert linear_warmup_rate > 0. and linear_warmup_rate < 1., '0 < linear_warmup_rate < 1.'

    warmup_steps = int(max_steps * linear_warmup_rate)  # n% of max_steps

    # Define the warmup scheduler
    def warmup_lambda(current_step):
        if current_step >= warmup_steps:
            return 1.0
        return float(current_step) / float(max(1, warmup_steps))

    # Create the warmup scheduler
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    # Create the cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(optimizer, max_steps - warmup_steps, eta_min=min_lr)

    # Combine the warmup and cosine annealing schedulers
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    return scheduler




def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a learning rate schedule that linearly increases the learning rate from
    0.0 to lr over ``num_warmup_steps``, then decreases to 0.0 on a cosine schedule over
    the remaining ``num_training_steps-num_warmup_steps`` (assuming ``num_cycles`` = 0.5).

    This is based on the Hugging Face implementation
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to
            schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of waves in the cosine schedule. Defaults to 0.5
            (decrease from the max value to 0 following a half-cosine).
        last_epoch (int): The index of the last epoch when resuming training. Defaults to -1

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """

    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        # cosine
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )

        cosine_lr_multiple = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        )
        return max(0.0, cosine_lr_multiple)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

