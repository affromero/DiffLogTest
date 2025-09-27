"""Decorator functions for unittest registration."""

import inspect
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from .logging import LoggingRich
from .utils import (
    is_unittest_mode,
    logfile_from_func,
    path_open,
)

_T = TypeVar("_T", bound=Callable[..., Any])


def _process_unittest_output(
    output: Any,
    log_file: str,
) -> str:
    """Process unittest output and logs into a formatted string.

    Arguments:
        output (Any): The output of the unittest.
        log_file (str): The path to the log file.

    Returns:
        str: The processed unittest output.

    Example:
        >>> _process_unittest_output("Hello World", "log.txt")

    Note:
        The function will raise a FileNotFoundError if the log file does not exist.

    """
    output_str = str(output)

    # merge the log file with the unittest log
    with path_open(log_file, "r") as f:
        log_content = f.read()

    # Return raw content - log processing will be done by UnitTests instance

    # Format the output string
    return (
        "# ----------------------------------------------------- #\n"
        "# ------------------ UNITTEST LOG --------------------- #\n"
        "# ----------------------------------------------------- #\n"
        f"{log_content}\n\n"
        "# ----------------------------------------------------- #\n"
        "# ------------------ UNITTEST OUTPUT ------------------ #\n"
        "# ----------------------------------------------------- #\n"
        f"{output_str}\n\n"
    )


def _setup_unittest(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    default_kwargs: dict[str, Any],
    logger: LoggingRich,
) -> tuple[str, inspect.BoundArguments]:
    """Set up unittest environment and bind arguments.

    Arguments:
        func (Callable[..., Any]): The function to be executed.
        args (tuple[Any, ...]): The arguments to be passed to the function.
        kwargs (dict[str, Any]): The keyword arguments to be passed to the function.
        default_kwargs (dict[str, Any]): The default keyword arguments to be used if not provided.
        logger (LoggingRich): The logger to be used for logging.

    Returns:
        tuple[str, inspect.BoundArguments]: A tuple containing the log file name and the bound arguments.

    """
    log_file = logfile_from_func(func)
    logger.unittest_logger(log_file)

    # Get the function's signature and bind arguments
    sig = inspect.signature(func)
    bound_args = sig.bind_partial(*args, **kwargs)

    # Update default values for missing parameters
    for name in sig.parameters:
        if name not in bound_args.arguments and name in default_kwargs:
            bound_args.arguments[name] = default_kwargs[name]

    return log_file, bound_args


def register_unittest(
    logger: LoggingRich,
    **default_kwargs: Any,
) -> Callable[[_T], _T]:
    """Decorate a function to register it as a unittest.

    Arguments:
        logger (LoggingRich): The logger to be used for logging.
        default_kwargs (dict[str, Any]): The default keyword arguments to be used if not provided.

    Returns:
        Callable[[_T], _T]: A decorator that can be used to decorate a function to register it as a unittest.

    """

    def decorator(func: _T) -> _T:
        import asyncio

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not is_unittest_mode():
                    return await func(*args, **kwargs)

                log_file, bound_args = _setup_unittest(
                    func, args, kwargs, default_kwargs, logger
                )

                try:
                    output = await func(*bound_args.args, **bound_args.kwargs)
                except Exception as e:
                    import traceback

                    traceback_error = traceback.format_exc()
                    logger.error(  # noqa: TRY400
                        f"Error in {func.__name__}: {e}\n{traceback_error}",
                    )
                    output = f"Error During Execution in {func.__name__}!"

                return _process_unittest_output(output, log_file)

            return async_wrapper  # type: ignore[return-value]

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_unittest_mode():
                return func(*args, **kwargs)

            log_file, bound_args = _setup_unittest(
                func, args, kwargs, default_kwargs, logger
            )

            try:
                output = func(*bound_args.args, **bound_args.kwargs)
            except Exception as e:
                import traceback

                traceback_error = traceback.format_exc()
                logger.error(  # noqa: TRY400
                    f"Error in {func.__name__}: {e}\n{traceback_error}",
                )
                output = f"Error During Execution in {func.__name__}!"

            return _process_unittest_output(output, log_file)

        return sync_wrapper  # type: ignore[return-value]

    return decorator
