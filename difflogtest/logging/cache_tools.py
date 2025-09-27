"""Cache tools."""

import functools
import inspect
import os
import sys
from collections.abc import Callable
from typing import Any, Protocol, TypeVar, cast, runtime_checkable

import jaxtyping
from beartype import beartype
from dotenv import dotenv_values
from sha256sum import sha256sum as _sha256sum

from difflogtest.utils.mode import is_unittest_mode
from difflogtest.utils.path import path_dotenv, path_expanduser, path_join

from .utils import get_logger

_T = TypeVar("_T")

logger = get_logger()


@runtime_checkable
class _CachedFunction(Protocol):
    def cache_info(self) -> Any: ...

    _original_qualname: str


_cached_functions: list[_CachedFunction] = []


def _get_cached_function_info(
    func: _CachedFunction | Callable[[_T], _T],
) -> str:
    """Get function info."""
    info = f" || {func.cache_info()}" if hasattr(func, "cache_info") else ""

    def _get_wrapped(
        func: _CachedFunction | Callable[[_T], _T],
    ) -> Callable[[_T], _T]:
        if hasattr(func, "__wrapped__"):
            return _get_wrapped(func.__wrapped__)
        return func  # type: ignore[return-value]

    wrapped_func = _get_wrapped(func)

    return f"{wrapped_func.__qualname__} (at {inspect.getfile(wrapped_func)}:{inspect.getsourcelines(wrapped_func)[1]}){info}"


def disable_lru_cache() -> bool:
    """Disable LRU cache."""
    return dotenv_values(path_dotenv()).get("DISABLE_LRU_CACHE") == "True"


def show_all_cache_info(
    func: _CachedFunction | Callable[[_T], _T] | None = None,
) -> None:
    """Show cache info for all cached functions."""
    msg = "-" * 30 + "\n"
    msg += "Cached functions:\n" if func is None else "Cached function: "
    if func is None:
        funcs = _cached_functions
    else:
        funcs = [_func for _func in _cached_functions if _func == func]
        if len(funcs) == 0:
            error = f"Function {func.__qualname__} not found in cache"  # type: ignore[union-attr]
            raise ValueError(error)
    for _func in funcs:
        # hits means the function was called with the same arguments
        # misses means the function was called with different arguments
        msg += _get_cached_function_info(_func) + "\n"
    msg += "-" * 30 + "\n"
    logger.debug(msg, stack_offset=2)


def lru_cache(
    maxsize: int | None = 128,
    *,
    max_misses: int = -1,
    print_cache_info: bool = True,
) -> Callable[[_T], _T]:
    """Decorate a function with functools.lru_cache, unless in unit test mode.

    This decorator wraps a function with functools.lru_cache, providing
        caching functionality for the function's results. However, if the code is
        running in unit test mode, the decorator is a no-op and does not provide caching.

    Arguments:
        maxsize (int | None): The maximum size of the cache. Defaults to 128.
        max_misses (int): The maximum number of misses before raising a cache miss error. Defaults to 1.
        print_cache_info (bool): Whether to print the cache info. Defaults to True.

    Returns:
        Callable: The decorated function. If in unit test mode, returns the
            original function without caching. Otherwise, returns the function
            wrapped with lru_cache, providing caching of its results.

    Example:
        >>> @lru_cache()
        >>> def example_function(arg1, arg2):
        >>>     return arg1 + arg2
    Note:
        This decorator is useful for preventing caching during unit tests,
            where repeated function calls with the same arguments are often needed.

    """

    def decorator(func: Callable[[_T], _T]) -> Callable[[_T], _T]:
        """Apply an LRU cache to the input function or return the function itself.

        This decorator function applies an LRU cache to the given function
            if the code is not running in unittest mode. If it is in
            unittest mode, the function is returned as is.

        Arguments:
            func (Callable[..., Any]): The function to be decorated.

        Returns:
            Callable[..., Any]: The decorated function with an applied LRU
                cache if not in unittest mode, or the original function if
                in unittest mode.

        Example:
            >>> @lru_cache()
            >>> def test_func(x, y):
            >>>     return x + y
        Note:
            This decorator is useful for optimizing functions that are
                called repeatedly with the same arguments, but should not be
                used in unittest mode to avoid cached results.

        """
        # Use cache when not in unit test mode
        if not disable_lru_cache():
            # Apply functools lru_cache
            cached_func = functools.lru_cache(maxsize=maxsize, typed=False)(
                func
            )
            cached_func._original_qualname = func.__qualname__  # type: ignore[attr-defined]
            _cached_functions.append(cast(_CachedFunction, cached_func))

            # Create a wrapper that calls show_all_cache_info() on execution
            @functools.wraps(cached_func)
            def cached_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Wrap that shows cache info on execution and preserves cache functionality."""
                output = cached_func(*args, **kwargs)
                info = cached_func.cache_info()
                if print_cache_info:
                    show_all_cache_info(cached_func)
                if max_misses != -1 and info.misses > max_misses:
                    msg = f"Function {func.__qualname__} has {info.misses} cache misses. This is too many. This is probably a bug. If not, update the max_misses parameter."
                    raise ValueError(msg)
                return output

            # Preserve the cache_info method and other attributes
            cached_wrapper.cache_info = cached_func.cache_info  # type: ignore[attr-defined]
            cached_wrapper._original_qualname = cached_func._original_qualname  # type: ignore[attr-defined]

            return cached_wrapper

        # No-operation decorator
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Act as a wrapper for another function, preserving its.

                metadata and allowing flexible argument passing.

            This function can be used to wrap another function,
                maintaining its metadata. It accepts any number of
                positional and keyword arguments, which are passed
                directly to the wrapped function.

            Arguments:
                *args (Any): Represents any number of positional
                    arguments that can be passed to the wrapped
                    function.
                **kwargs (Any): Represents any number of keyword
                    arguments that can be passed to the wrapped
                    function.

            Returns:
                Any: Returns the result of calling the wrapped function
                    with the provided arguments and keyword arguments.

            Example:
                >>> wrapped_function =
                    wrapper_function(original_function, arg1, arg2,
                    keyword_arg1=value1)

            Note:
                The wrapped function and its arguments are not specified
                    until the wrapper function is called.

            """
            if not is_unittest_mode():
                logger.warning(
                    f"{_get_cached_function_info(func)} has @lru_cache but it is disabled!"
                )
            return func(*args, **kwargs)

        return wrapper

    return cast("Callable[[_T], _T]", decorator)


def jaxtyped(typechecker: Callable[[_T], _T] = beartype) -> Callable[[_T], _T]:
    """Decorate a function with jaxtyping, unless in unit test mode.

    This decorator wraps a function with jaxtyping, providing
        type checking for the function's arguments and return value.

    """

    def decorator(func: Callable[[_T], _T]) -> Callable[[_T], _T]:
        """Apply jaxtyping to the input function or return the function.

        This decorator function applies jaxtyping to the given function
            if the code is not running in unittest mode. If it is in
            unittest mode, the function is returned as is.
        """
        return jaxtyping.jaxtyped(typechecker=typechecker)(func)

    return cast(Callable[[_T], _T], decorator)


def get_cache_dir() -> str:
    """Locate a platform-appropriate cache directory for flit to use.

    This function identifies the appropriate cache directory for the
        specified platform and app. It does not ensure
    that the cache directory exists.

    Arguments:
        platform (str): The platform for which the cache directory is to be
            located.
        app (str): The application for which the cache directory is to be
            located.
        flit (bool | None): A flag indicating if flit is to be used.
            Defaults to None.

    Returns:
        str: The path of the located cache directory.

    Example:
        >>> locate_cache_dir("Windows", "flit", flit=True)

    Note:
        The function does not create the cache directory, it only locates
            the appropriate directory for the given platform and app.

    """
    # Linux, Unix, AIX, etc.
    if os.name == "posix" and sys.platform != "darwin":
        # use ~/.cache if empty OR not set
        cache_dir = os.environ.get("XDG_CACHE_HOME", None) or path_expanduser(
            "~/.cache"
        )

    # Mac OS
    elif sys.platform == "darwin":
        cache_dir = path_join(path_expanduser("~"), "Library/Caches")

    # Windows
    else:
        cache_dir = os.environ.get("LOCALAPPDATA", None) or path_expanduser(  # type: ignore[arg-type]
            "~\\AppData\\Local",
        )

    return cache_dir


@jaxtyped(typechecker=beartype)
def sha256sum(filename: str) -> str:
    """Calculate the SHA-256 hash of a file and return the first 8 characters.

    This function takes a filename as an argument, calculates the SHA-256
        hash of the file, and returns the first 8 characters of the hash.

    Arguments:
        filename (str): A string representing the name of the file for which
            the SHA-256 hash needs to be calculated.

    Returns:
        str: A string containing the first 8 characters of the SHA-256 hash
            of the file.

    Example:
        >>> calculate_file_hash("example.txt")

    Note:
        The file must exist in the current working directory or a full path
            must be provided.

    """
    return _sha256sum(filename)[:8]
