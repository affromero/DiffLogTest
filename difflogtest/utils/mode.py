"""Environment and mode management functions."""

import dotenv


def enable_debug_mode() -> None:
    """Enable the debug mode."""
    dotenv.set_key(dotenv.find_dotenv(), "LOG_DEBUG", "True")


def disable_debug_mode() -> None:
    """Disable the debug mode."""
    dotenv.set_key(dotenv.find_dotenv(), "LOG_DEBUG", "False")


def set_unittest_mode(
    *, disable_lru_cache: bool = True, cuda_deterministic: bool = True
) -> None:
    """Set the unittest mode by updating environment variables in the .env file.

    Args:
        disable_lru_cache: Whether to disable LRU cache.
        cuda_deterministic: Whether to enable deterministic CUDA operations.

    """
    env_path = dotenv.find_dotenv()
    if env_path == "":
        msg = (
            "No .env file found. A .env file is required to store environment variables "
            "that control unittest mode and other runtime behaviors. Please create a .env file "
            "in your project root to proceed."
        )
        raise FileNotFoundError(msg)

    dotenv.set_key(env_path, "IS_UNITTEST_MODE", "True")
    dotenv.set_key(
        env_path,
        "CUDA_DETERMINISTIC",
        "True" if cuda_deterministic else "False",
    )
    dotenv.set_key(env_path, "TQDM_DISABLE", "True")
    dotenv.set_key(
        env_path,
        "DISABLE_LRU_CACHE",
        "True" if disable_lru_cache else "False",
    )


def unset_unittest_mode() -> None:
    """Unset the unittest mode."""
    dotenv.set_key(dotenv.find_dotenv(), "IS_UNITTEST_MODE", "False")
    dotenv.set_key(dotenv.find_dotenv(), "CUDA_DETERMINISTIC", "False")
    dotenv.set_key(dotenv.find_dotenv(), "TQDM_DISABLE", "False")
    dotenv.set_key(dotenv.find_dotenv(), "DISABLE_LRU_CACHE", "False")


def is_unittest_mode() -> bool:
    """Check if the unittest mode is enabled."""
    return dotenv.get_key(dotenv.find_dotenv(), "IS_UNITTEST_MODE") == "True"
