"""Log content replacement utilities."""

import re

from pydantic.dataclasses import dataclass


@dataclass
class LogReplacement:
    """Configuration for log content replacements."""

    name: str
    """Name/identifier for this replacement rule."""

    pattern: str
    """Regex pattern or literal string to match."""

    replacement: str
    """String to replace matches with."""

    use_regex: bool = True
    """Whether to use regex matching (True) or literal string replacement (False)."""

    def apply(self, content: str) -> str:
        """Apply this replacement to the given content."""
        if self.use_regex:
            return re.sub(self.pattern, self.replacement, content)
        return content.replace(self.pattern, self.replacement)

    @classmethod
    def create_defaults(cls) -> list["LogReplacement"]:
        """Create the default set of log replacements.

        Returns:
            list[LogReplacement]: Default replacement rules.

        """
        return [
            # Timestamps in various formats
            cls(
                name="timestamps",
                # Patterns are applied in the order they appear in the list.
                # This pattern matches various timestamp and duration formats.
                pattern=(
                    r"\d{4}-\d{2}-\d{2}[T_ ]\d{2}[:-]\d{2}[:-]\d{2}(?:\.\d{6}[+-]\d{2}:\d{2})?"  # ISO, underscore, or space-separated timestamps
                    r"|\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z"  # UTC Zulu timestamps
                    r"|\d{1,2}:\d{2}:\d{2}"  # Short time (H:MM:SS or HH:MM:SS)
                    r"|\d{2}d : \d{2}h : \d{2}m : \d{2}s"  # Duration format
                ),
                replacement="<TIMESTAMP>",
                use_regex=True,
            ),
            # Memory addresses (hexadecimal)
            cls(
                name="memory_addresses",
                pattern=r"0x[0-9a-fA-F]+",
                replacement="<MEMORY_ADDRESS>",
                use_regex=True,
            ),
            # Home directory paths
            cls(
                name="home_directory",
                pattern=r"/home/[^/\s]+",
                replacement="<HOME_DIR>",
                use_regex=True,
            ),
            # HuggingFace snapshot hashes
            cls(
                name="huggingface_snapshot",
                pattern=r"/snapshots/[a-f0-9]{40}",
                replacement="/snapshots/<HUGGINGFACE_SNAPSHOT>",
                use_regex=True,
            ),
            # Git commit hashes
            cls(
                name="commit_hash",
                pattern=r"\b[0-9a-f]{40}\b",
                replacement="<COMMIT_HASH>",
                use_regex=True,
            ),
            # UUID hashes
            cls(
                name="uuid_hash",
                pattern=r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                replacement="<UUID_HASH>",
                use_regex=True,
            ),
            # Temporary/temporal files
            cls(
                name="temporal_files",
                pattern=r"/tmp/[^/\s]+|\.tmp$",  # noqa: S108
                replacement="<TEMPORAL_FILE>",
                use_regex=True,
            ),
        ]


def process_log_content(
    content: str, replacements: list[LogReplacement]
) -> str:
    """Process log content using the provided replacement rules.

    Arguments:
        content (str): The content to process.
        replacements (list[LogReplacement]): List of replacement rules to apply.

    Returns:
        str: The processed content.

    """
    for replacement in replacements:
        content = replacement.apply(content)
    return content


def add_log_replacement(
    replacements: list[LogReplacement], replacement: LogReplacement
) -> None:
    r"""Add a custom log replacement rule to the list.

    Arguments:
        replacements: The list of replacements to modify.
        replacement: A LogReplacement instance defining the pattern and replacement.

    Example:
        >>> from difflogtest import LogReplacement
        >>> replacements = LogReplacement.create_defaults()
        >>> custom_replacement = LogReplacement(
        ...     name="custom_pattern",
        ...     pattern=r"my_custom_\w+",
        ...     replacement="<CUSTOM>",
        ...     use_regex=True,
        ... )
        >>> add_log_replacement(replacements, custom_replacement)

    """
    replacements.append(replacement)


def remove_log_replacement(
    replacements: list[LogReplacement], name: str
) -> None:
    """Remove a log replacement rule by name from the list.

    Arguments:
        replacements: The list of replacements to modify.
        name: The name of the replacement rule to remove.

    """
    replacements[:] = [r for r in replacements if r.name != name]


def clear_log_replacements(replacements: list[LogReplacement]) -> None:
    """Clear all log replacement rules from the list.

    Arguments:
        replacements: The list of replacements to clear.

    """
    replacements.clear()


def reset_log_replacements(replacements: list[LogReplacement]) -> None:
    """Reset log replacements to defaults.

    Arguments:
        replacements: The list of replacements to reset.

    """
    replacements.clear()
    replacements.extend(LogReplacement.create_defaults())
