"""Command line interface for difflogtest."""

import tyro

from .core import UnitTests
from .utils.mode import unset_unittest_mode


def entrypoint() -> None:
    """Run the unit tests."""
    try:
        tyro.cli(
            UnitTests,
            description="Run all registered difflogtest unittests with rich CLI help and argument completion.",
            show_defaults=True,
            print_help_on_error=True,
        )
    finally:
        unset_unittest_mode()


if __name__ == "__main__":
    entrypoint()
