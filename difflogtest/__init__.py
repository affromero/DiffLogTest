"""Difflogtest - A unittest framework for comparing outputs with logged expectations."""

from .core import UnitTests
from .decorators import register_unittest
from .logging import (
    DEFAULT_VERBOSITY,
    LoggingRich,
    get_logger,
    seed_everything,
    wait_seconds_bar,
)
from .logging.cache_tools import (
    disable_lru_cache,
    get_cache_dir,
    jaxtyped,
    lru_cache,
    sha256sum,
)
from .utils import (
    LogReplacement,
    add_log_replacement,
    clear_log_replacements,
    is_unittest_mode,
    process_log_content,
    remove_log_replacement,
    reset_log_replacements,
)

__all__ = [
    "DEFAULT_VERBOSITY",
    "LogReplacement",
    "LoggingRich",
    "UnitTests",
    "add_log_replacement",
    "clear_log_replacements",
    "disable_lru_cache",
    "get_cache_dir",
    "get_logger",
    "is_unittest_mode",
    "jaxtyped",
    "lru_cache",
    "process_log_content",
    "register_unittest",
    "remove_log_replacement",
    "reset_log_replacements",
    "seed_everything",
    "sha256sum",
    "wait_seconds_bar",
]
