"""Utility functions for difflogtest."""

from .mode import (
    disable_debug_mode,
    enable_debug_mode,
    is_unittest_mode,
    set_unittest_mode,
    unset_unittest_mode,
)
from .path import (
    is_file_changed,
    logfile_from_func,
    path_mkdir,
    path_open,
    path_relative_to,
)
from .replacements import (
    LogReplacement,
    add_log_replacement,
    clear_log_replacements,
    process_log_content,
    remove_log_replacement,
    reset_log_replacements,
)
from .strings import (
    get_elapsed_time,
)

__all__ = [
    "LogReplacement",
    "add_log_replacement",
    "clear_log_replacements",
    "disable_debug_mode",
    "enable_debug_mode",
    "get_elapsed_time",
    "is_file_changed",
    "is_unittest_mode",
    "logfile_from_func",
    "path_mkdir",
    "path_open",
    "path_relative_to",
    "process_log_content",
    "remove_log_replacement",
    "reset_log_replacements",
    "set_unittest_mode",
    "unset_unittest_mode",
]
