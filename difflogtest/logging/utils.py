"""Helper functions for logging."""

import os
import random
import time
from functools import lru_cache

import numpy as np
import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, TimeRemainingColumn

from .core import DEFAULT_VERBOSITY, LoggingRich

load_dotenv()

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min

SEED_EVERYTHING = os.getenv("SEED_EVERYTHING", "-1")
CUDA_DETERMINISTIC = os.getenv("CUDA_DETERMINISTIC", "False")


def seed_everything(
    seed: int | None = None,
    *,
    workers: bool = False,
    verbose: bool = True,
    cuda_deterministic: bool = True,
) -> int:
    """Set the seed for pseudo-random number generators in torch, numpy, and random."""
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = 0
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = 0
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        seed = 0

    if verbose:
        print(f"Seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


if SEED_EVERYTHING != "-1":
    seed_everything(
        int(SEED_EVERYTHING),
        verbose=False,
        cuda_deterministic=CUDA_DETERMINISTIC == "True",
    )


@lru_cache(maxsize=128)
def get_logger(verbosity: dict[str, bool] = DEFAULT_VERBOSITY) -> LoggingRich:
    """Create a LoggingRich object with a specified verbosity level."""
    return LoggingRich(verbosity=verbosity)


def wait_seconds_bar(total_seconds: int, /, description: str) -> None:
    """Display a progress bar for a specified number of seconds."""
    # rich progress bar with time
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=Console(),
    ) as progress:
        task = progress.add_task(description, total=total_seconds)
        for _ in range(total_seconds):
            time.sleep(1)
            progress.update(task, advance=1)
