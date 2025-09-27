"""Helper functions for path operations."""

import glob
import os
import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any, Literal

import dotenv
import wget
from git import Repo
from natsort import natsorted, ns
from rich import print as pprint


def get_suffix(path: str | Path) -> str:
    """Get the suffix of a path."""
    return Path(path).suffix


def path_replace_suffix(path: str | Path, suffix: str) -> str:
    """Replace the suffix of a path."""
    return str(Path(path).with_suffix(suffix))


def path_rstrip(path: str | Path, suffix: str) -> str:
    """Remove the suffix from a path."""
    return str(path).rstrip(suffix)


def path_resolve(path: str | Path) -> str:
    """Resolve a path."""
    return str(Path(path).resolve())


def path_rename(path: str | Path, new_name: str | Path) -> None:
    """Rename a path."""
    Path(path).rename(new_name)


def path_dotenv() -> str:
    """Get the path to the .env file."""
    file_env = path_join(path_cwd(), ".env")
    if not path_exists(file_env):
        msg = f"File {file_env} not found"
        raise FileNotFoundError(msg)
    return file_env


def keep_local_data() -> bool:
    """Check if local data should be kept."""
    return dotenv.get_key(path_dotenv(), "KEEP_LOCAL_DATA") == "True"


def path_home() -> str:
    """Get the home directory."""
    return str(Path.home())


def path_getmtime(path: str | Path) -> float:
    """Get the modification time of a path."""
    return Path(path).stat().st_mtime


def path_join(*paths: str | Path) -> str:
    """Join multiple paths together.

    This function joins multiple paths together using the '/' separator.

    Arguments:
        *paths: str or Path. The paths to join.
        out (str): The output type. Default is 'str'.

    Returns:
        str or Path: The joined path.

    Example:
        >>> path_join("path", "to", "file", out="str")
        'path/to/file'

    Note:
        This function is useful in joining multiple paths together.

    """
    return "/".join([str(p) for p in paths])


def path_cwd() -> str:
    """Get the current working directory."""
    return str(Path.cwd())


def path_dirname(path: str | Path) -> str:
    """Get the directory name of a path.

    This function returns the directory name of the specified path.

    Arguments:
        path (str | Path): The path to get the directory name from.

    Returns:
        str: The directory name of the specified path.

    Example:
        >>> dirname("path/to/file")
        'path/to'

    Note:
        This function is useful in getting the directory name of a path.

    """
    return str(Path(path).parent)


def path_symlink(
    src: str | Path,
    dst: str | Path,
    *,
    ignore_existing: bool = False,
) -> None:
    """Create a symbolic link. If the destination already exists, it will be ignored.

    This function creates a symbolic link at the specified destination.

    Arguments:
        src (str | Path): The source path to create the symbolic link from.
        dst (str | Path): The destination path to create the symbolic link to.
        ignore_existing (bool): Whether to ignore the existing destination. Default is False.

    Returns:
        None: This function does not return any value.

    Example:
        >>> path_symlink("path/to/src", "path/to/dst")

    Note:
        This function is useful in creating a symbolic link.

    """
    if ignore_existing and path_exists(dst):
        return
    Path(dst).symlink_to(Path(src).resolve())


def path_relative_to(path: str | Path, base: str | Path) -> str:
    """Get the relative path of a path.

    This function returns the relative path of the specified path with respect to the base path.

    Arguments:
        path (str | Path): The path to get the relative path from.
        base (str | Path): The base path to get the relative path to.

    Returns:
        str: The relative path of the specified path with respect to the base path.

    Example:
        >>> path_relative_to("path/to/file", "path/to/")
        'file'

    Note:
        This function is useful in getting the relative path of a path.

    """
    return str(Path(path).absolute().relative_to(Path(base).absolute()))


def path_expanduser(path: str | Path) -> str:
    """Expand a path to include the user's home directory.

    This function expands the specified path to include the user's home
        directory.

    Arguments:
        path (str | Path): The path to expand.

    Returns:
        str: The expanded path.

    Example:
        >>> path_expanduser("~/Documents")
        '/home/user/Documents'

    Note:
        This function is useful in expanding a path to include the user's home directory.

    """
    return str(Path(path).expanduser())


def path_basename(path: str | Path) -> str:
    """Get the base name of a path.

    This function returns the base name of the specified path.

    Arguments:
        path (str | Path): The path to get the base name from.

    Returns:
        str: The base name of the specified path.

    Example:
        >>> basename("path/to/file.format")
        'file.format'

    Note:
        This function is useful in getting the base name of a path.

    """
    return Path(path).name


def path_stem(path: str | Path) -> str:
    """Get the stem of a path.

    This function returns the stem of the specified path.

    Arguments:
        path (str | Path): The path to get the stem from.

    Returns:
        str: The stem of the specified path.

    Example:
        >>> stem("path/to/file.format")
        'file'

    Note:
        This function is useful in getting the stem of a path.

    """
    return Path(path).stem


def path_exists(path: str | Path) -> bool:
    """Check if a path exists.

    This function checks if the specified path exists.

    Arguments:
        path (str | Path): The path to check for existence.

    Returns:
        bool: True if the path exists, False otherwise.

    Example:
        >>> exists("path/to/file")
        True

    Note:
        This function is useful in checking if a path exists.

    """
    return Path(path).exists()


def path_is_s3(path: str | Path) -> bool:
    """Check if a path is in S3."""
    return str(path).startswith("s3://")


def path_dir_empty(path: str | Path) -> bool:
    """Check if a directory is empty.

    This function checks if the specified directory is empty.

    Arguments:
        path (str | Path): The path to check if it is a directory.

    Returns:
        bool: True if the directory is empty, False otherwise.

    Example:
        >>> path_dir_empty("path/to/dir")

    Note:
        This function is useful in checking if a directory is empty.

    """
    if not path_exists(path):
        return False
    return not any(Path(path).iterdir())


def path_exists_and_not_empty(path: str | Path) -> bool:
    """Check if a path exists and is not empty."""
    return path_exists(path) and not path_dir_empty(path)


def path_is_dir(path: str | Path) -> bool:
    """Check if a path is a directory.

    This function checks if the specified path is a directory.

    Arguments:
        path (str | Path): The path to check if it is a directory.

    Returns:
        bool: True if the path is a directory, False otherwise.

    Example:
        >>> is_dir("path/to/dir")
        True

    Note:
        This function is useful in checking if a path is a directory.

    """
    return Path(path).is_dir()


def path_is_file(path: str | Path) -> bool:
    """Check if a path is a file.

    This function checks if the specified path is a file.

    Arguments:
        path (str | Path): The path to check if it is a file.

    Returns:
        bool: True if the path is a file, False otherwise.

    Example:
        >>> is_file("path/to/file")
        True

    Note:
        This function is useful in checking if a path is a file.

    """
    return Path(path).is_file() or Path(path).is_symlink()


def path_is_image_file(path: str | Path) -> bool:
    """Check if a path is an image file."""
    return get_suffix(path).lower() in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".heic",
        ".heif",
        ".webp",
    ]


def path_is_video_file(path: str | Path) -> bool:
    """Check if a path is a video file."""
    return get_suffix(path).lower() in [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    ]


def path_absolute(path: str | Path) -> str:
    """Get the absolute path of a path.

    This function returns the absolute path of the specified path.

    Arguments:
        path (str | Path): The path to get the absolute path from.

    Returns:
        str: The absolute path of the specified path.

    Example:
        >>> path_absolute("path/to/file")
        '/path/to/file'

    Note:
        This function is useful in getting the absolute path of a path.

    """
    return str(Path(path).absolute())


def path_mkdir(
    path: str | Path,
    *,
    parents: bool = False,
    exist_ok: bool = False,
) -> None:
    """Create a directory.

    This function creates a directory at the specified path.

    Arguments:
        path (str | Path): The path to create the directory.
        parents (bool): Whether to create parent directories. Default is False.
        exist_ok (bool): Whether to raise an error if the directory exists.
            Default is False.

    Returns:
        None: This function does not return any value.

    Example:
        >>> mkdir("path/to/dir", parents=True, exist_ok=True)

    Note:
        This function is useful in creating a directory.

    """
    Path(path).mkdir(parents=parents, exist_ok=exist_ok)


def path_remove(
    path: str | Path, *, non_exist_ok: bool = False, verbose: bool = False
) -> None:
    """Remove a file or directory.

    This function removes the specified file or directory.

    Arguments:
        path (str | Path): The path to remove.
        non_exist_ok (bool): Whether to raise an error if the file or directory does not exist.
            Default is False.
        verbose (bool): Whether to print messages. Default is False.

    Returns:
        None: This function does not return any value.

    Example:
        >>> remove("path/to/file")

    Note:
        This function is useful in removing a file or directory.

    """
    if verbose:
        pprint(f"!!![bold red]Removing {path}")
    Path(path).unlink(missing_ok=non_exist_ok)


def path_copy(src: str | Path, dst: str | Path) -> str:
    """Copy a file or directory.

    This function copies the specified file or directory to the destination.

    Arguments:
        src (str | Path): The source file or directory to copy.
        dst (str | Path): The destination file or directory to copy to.

    Returns:
        None: This function does not return any value.

    Example:
        >>> copy("path/to/src", "path/to/dst")

    Note:
        This function is useful in copying a file or directory.

    """
    return str(shutil.copy(src, dst))


def path_copy_dir(src: str | Path, dst: str | Path) -> str:
    """Copy a directory.

    This function copies the specified directory to the destination.

    Arguments:
        src (str | Path): The source directory to copy.
        dst (str | Path): The destination directory to copy to.

    Returns:
        None: This function does not return any value.

    Example:
        >>> copy_dir("path/to/src", "path/to/dst")

    Note:
        This function is useful in copying a directory.

    """
    path_mkdir(path_dirname(dst), exist_ok=True, parents=True)
    return str(shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True))


def path_move(src: str | Path, dst: str | Path) -> str:
    """Move a file or directory.

    This function moves the specified file or directory to the destination.

    Arguments:
        src (str | Path): The source file or directory to move.
        dst (str | Path): The destination file or directory to move to.

    Returns:
        None: This function does not return any value.

    Example:
        >>> move("path/to/src", "path/to/dst")

    Note:
        This function is useful in moving a file or directory.

    """
    return str(shutil.move(src, dst))


def path_read_text(path: str | Path) -> str:
    """Read text from a file.

    This function reads text from the specified file.

    Arguments:
        path (str | Path): The path to read text from.

    Returns:
        str: The text read from the file.

    Example:
        >>> read_text("path/to/file")
        'text'

    Note:
        This function is useful in reading text from a file.

    """
    return Path(path).read_text()


def path_open(
    path: str | Path,
    mode: str = "r",
    *,
    newline: str | None = None,
    encoding: str | None = None,
) -> IO[Any]:
    """Open a file.

    This function opens the specified file.

    Arguments:
        path (str | Path): The path to open.
        mode (str): The mode to open the file in. Default is 'r'.
        newline (str | None): The newline character to use. Default is None.
        encoding (str | None): The encoding to use. Default is None.

    Returns:
        file: The opened file.

    Example:
        >>> open("path/to/file", "r")

    Note:
        This function is useful in opening a file.

    """
    return Path(path).open(mode, newline=newline, encoding=encoding)


def path_write_text(path: str | Path, text: str) -> None:
    """Write text to a file.

    This function writes text to the specified file.

    Arguments:
        path (str | Path): The path to write text to.
        text (str): The text to write to the file.

    Returns:
        None: This function does not return any value.

    Example:
        >>> write_text("path/to/file", "text")

    Note:
        This function is useful in writing text to a file.

    """
    Path(path).write_text(text)


def path_startswith(path: str | Path, start: str) -> bool:
    """Check if a path starts with a string.

    This function checks if the specified path starts with the specified string.

    Arguments:
        path (str | Path): The path to check.
        start (str): The string to check if the path starts with.

    Returns:
        bool: True if the path starts with the string, False otherwise.

    Example:
        >>> startswith("path/to/file", "path")

    Note:
        This function is useful in checking if a path starts with a string.

    """
    return str(path).startswith(start)


def path_endswith(path: str | Path, end: str) -> bool:
    """Check if a path ends with a string.

    This function checks if the specified path ends with the specified string.

    Arguments:
        path (str | Path): The path to check.
        end (str): The string to check if the path ends with.

    Returns:
        bool: True if the path ends with the string, False otherwise.

    Example:
        >>> endswith("path/to/file", "file")

    Note:
        This function is useful in checking if a path ends with a string.

    """
    return str(path).endswith(end)


def path_glob(path: str | Path, *, sort: bool = True) -> list[str]:
    """Get a list of paths that match a pattern.

    This function returns a list of paths that match the specified pattern.

    Arguments:
        path (str | Path): The path to get a list of paths from.
        sort (bool): Whether to sort the paths. Default is False.

    Returns:
        list[str]: A list of paths that match the pattern.

    Example:
        >>> glob("path/to/*.txt")

    Note:
        This function is useful in getting a list of paths that match a pattern.

    """
    return sorted(glob.glob(str(path))) if sort else glob.glob(str(path))  # noqa: PTH207


def path_rglob(
    path: str | Path, *, pattern: str, sort: bool = True
) -> list[str]:
    """Get a list of paths that match a pattern recursively.

    This function returns a list of paths that match the specified pattern recursively.

    Arguments:
        path (str | Path): The path to get a list of paths from.
        pattern (str): The pattern to match.
        sort (bool): Whether to sort the paths. Default is True.

    Returns:
        list[str]: A list of paths that match the pattern recursively.

    Example:
        >>> rglob("path/to/**/*.txt")

    Note:
        This function is useful in getting a list of paths that match a pattern recursively.

    """
    if "*" in str(path):
        msg = "Path cannot contain * - use path_glob instead"
        raise ValueError(msg)
    path_rglob = [str(i) for i in Path(path).rglob(pattern)]
    return sorted(path_rglob) if sort else list(path_rglob)


def path_abs(path: str | Path) -> str:
    """Get the absolute path of a path.

    This function returns the absolute path of the specified path.

    Arguments:
        path (str | Path): The path to get the absolute path from.

    Returns:
        str: The absolute path of the specified path.

    Example:
        >>> abs("path/to/file")
        '/path/to/file'

    Note:
        This function is useful in getting the absolute path of a path.

    """
    return str(Path(path).absolute())


def path_remove_dir(
    path: str | Path,
    *,
    verbose: bool = False,
    non_exist_ok: bool = False,
    only_files: bool = False,
) -> None:
    """Remove a directory.

    This function removes the specified directory.

    Arguments:
        path (str | Path): The directory to remove.
        verbose (bool): Whether to print messages. Default is False.
        non_exist_ok (bool): Whether to raise an error if the directory does not exist.
            Default is False.
        only_files (bool): Whether to remove only the files in the directory, but not the directory itself. Default is False.

    Returns:
        None: This function does not return any value.

    Example:
        >>> remove_dir("path/to/dir")

    Note:
        This function is useful in removing a directory.

    """
    if path_is_dir(path):
        for file in path_rglob(path, pattern="*"):
            if verbose:
                pprint(f"!!![bold red]Removing {file}")
            if path_is_file(file):
                path_remove(file)
        if not only_files:
            shutil.rmtree(path)
    elif not non_exist_ok:
        msg = f"{path} is not a directory"
        raise ValueError(msg)


def path_listdir(
    path: str | Path,
    /,
    *,
    include_hidden: bool = False,
    include_private: bool = False,
) -> list[str]:
    """List the contents of a directory.

    This function returns a list of the contents of the specified directory.

    Arguments:
        path (str | Path): The path to list the contents of.
        include_hidden (bool): Whether to include hidden files and directories.
            Default is False.
        include_private (bool): Whether to include private files and directories.
            Default is False.

    Returns:
        list[str]: A list of the contents of the specified directory.

    Example:
        >>> listdir("path/to/dir")

    Note:
        This function is useful in listing the contents of a directory.

    """
    dirs = [path_relative_to(i, path) for i in Path(path).iterdir()]
    if not include_hidden:
        dirs = [i for i in dirs if not i.startswith(".")]
    if not include_private:
        dirs = [i for i in dirs if not i.startswith("__")]
    return dirs


def path_newest_dir(_dir: str | Path, /) -> list[str]:
    """Retrieve the most recently modified directories within a specified directory.

    Arguments:
        _dir (str | Path): The path of the directory to be inspected.

    Returns:
        list[str]: A list of directory paths, sorted from newest to oldest by modification time.

    Example:
        >>> path_newest_dir("/home/user/Documents")
        ['/home/user/Documents/dir1', '/home/user/Documents/dir2']

    Note:
        If the specified directory does not contain any directories, a ValueError will be raised.

    """
    subdirs = path_listdir(_dir)
    subdirs = [path_join(_dir, d) for d in subdirs]
    dirs = [Path(d) for d in subdirs if path_is_dir(d)]
    if len(dirs) == 0:
        msg = f"No directory in {_dir}"
        raise ValueError(msg)
    dirs.sort(key=lambda x: path_stat(x).st_mtime)
    return [str(d) for d in dirs[::-1]]


def path_newest_file(
    _dir: str | Path,
    /,
    *,
    pattern: str,
) -> str:
    """Retrieve the most recently modified file within a specified directory.

    Arguments:
        _dir (str | Path): The directory to search for the most recently modified file.
        pattern (str): The pattern to match the files.

    Returns:
        str: The path to the most recently modified file.

    """
    files = [f for f in Path(_dir).glob(pattern) if f.is_file()]
    files.sort(key=lambda x: path_stat(x).st_mtime)
    return str(files[-1])


def path_stat(path: str | Path, /) -> os.stat_result:
    """Get the stat of a path."""
    return Path(path).stat()


def path_download_and_extract_tar(
    url: str,
    path: str | Path,
    *,
    verbose: bool = True,
) -> None:
    """Download and extract a tar file.

    This function downloads a tar file from the specified URL and extracts it to the specified path.

    Arguments:
        url (str): The URL to download the tar file from.
        path (str | Path): The path to extract the tar file to.
        verbose (bool): Whether to print messages. Default is True.

    Returns:
        None: This function does not return any value.

    Example:
        >>> download_and_extract_tar("https://example.com/file.tar", "path/to/dir")

    Note:
        This function is useful in downloading and extracting a tar file.

    """
    if path_exists(path):
        msg = f"{path=} already exists"
        raise ValueError(msg)
    path_suffix = path_join(path_dirname(path), path_basename(url))
    if not path_exists(path_suffix):
        if verbose:
            pprint(f"Downloading {url} to {path_suffix}")
        wget.download(url, path_suffix)
    with tarfile.open(path_suffix) as tar:
        tar.extractall(path=path_dirname(path), filter="data")
    # path_remove(path_suffix)
    if verbose:
        pprint(f"Downloaded and extracted {url} to {path}")


def path_from_pattern(
    basedir: str | Path,
    /,
    *,
    pattern: list[str],
    recursive: bool = True,
    ignore: list[str] | None = None,
    output_type: Literal["Path", "str"] = "Path",
) -> list[Path] | list[str]:
    """Search for files in a given directory based on specified patterns.

    Arguments:
        basedir (Union[str, Path]): The base directory where files will be
            searched. It can be a string or a Path object.
        pattern (List[str]): A list of file patterns to filter files. Each
            pattern is a string.
        recursive (bool): A flag to indicate whether to search files
            recursively. Defaults to True.
        ignore (Optional[List[str]]): A list of patterns to ignore while
            filtering files. Each pattern is a string. Defaults to None.
        output_type (Literal['Path', 'str']): The type of output to return,
            either 'Path' or 'str'. Defaults to 'Path'.

    Returns:
        Union[List[Path], List[str]]: A list of paths or strings based on
            the output type specified. If 'Path' is specified, it returns a
            list of Path objects. If 'str' is specified, it returns a list
            of strings.

    Example:
        >>> search_files('/home/user', ['*.txt', '*.docx'], recursive=True,
            ignore=['*.tmp'], output_type='str')

    Note:
        The function uses glob patterns for file filtering.

    """
    image_list = []
    if isinstance(basedir, str):
        basedir = Path(basedir)
    for _pattern in pattern:
        patterns = _pattern.split(" & ") if " & " in _pattern else [_pattern]
        # if pattern is "pattern1 & pattern2", then the image must contain both patterns

        if recursive:
            current_files = [
                i
                for i in sorted(basedir.rglob("*"))
                if i.is_file() and all(p in str(i) for p in patterns)
            ]
        else:
            current_files = [
                i
                for i in sorted(basedir.glob("*"))
                if i.is_file() and all(p in str(i) for p in patterns)
            ]
        image_list.extend(current_files)
    if ignore is not None:
        image_list = [
            i for i in image_list if not any(ig in str(i) for ig in ignore)
        ]
    image_list = natsorted(image_list, alg=ns.IGNORECASE)
    if output_type == "str":
        return [str(i) for i in image_list]
    return image_list


def is_file_changed(
    file_path: str,
    *,
    ignore_if_is_new: bool = True,
) -> bool:
    """Determine if Git detects any modifications in the specified file.

    This function checks if there are any unstaged changes in the given
        file as per Git's tracking.
    If the file is new and the 'ignore_if_is_new' parameter is set to
        True, the function will return False regardless of the file
        status.

    Arguments:
        file_path (str): The absolute or relative path to the file to be
            checked.
        ignore_if_is_new (bool): If set to True, the function will
            return False for new files regardless of their status.
            Defaults to False.

    Returns:
        bool: Returns True if Git detects changes in the file and it's
            not a new file (or new files are not being ignored). Returns
            False otherwise.

    Example:
        >>> check_git_changes("/path/to/file", False)

    Note:
        This function assumes that the current working directory is a
        Git repository.

    """
    # Get the absolute path to ensure correctness
    rel_file_path = path_relative_to(
        path_absolute(file_path),
        path_cwd(),
    )

    # Find the git repository based on the file's location
    repo = Repo(search_parent_directories=True)

    if ignore_if_is_new and rel_file_path in repo.untracked_files:
        return False

    change_diff: str = repo.git.diff(rel_file_path)

    # If changes are detected by Git, `change_diff` will not be empty
    return bool(change_diff)


def logfile_from_func(
    func: Callable[..., Any], *, log_folder: str = "unittest_logs"
) -> str:
    """Retrieve the log file path for a function decorated with.

        @register_unittest.

    This function is used to get the path of the log file corresponding to a
        function that has been decorated with the @register_unittest
        decorator. The log file contains the output of the unit test for the
        decorated function.

    Arguments:
        func (Callable): The function decorated with @register_unittest.
            This is the function for which the log file path is being
            retrieved.
        log_folder (str): The folder to store the log file. Default is "unittest_logs".

    Returns:
        str: The path to the log file. This is a string representing the
            location of the log file on the file system.

    Example:
        >>> get_log_file_path(my_decorated_function)

    Note:
        The function passed as argument must be decorated with
            @register_unittest, otherwise, the function will not behave as
            expected.

    """
    relative_dir = path_relative_to(
        path_dirname(func.__code__.co_filename),
        path_cwd(),
    )
    name_file = path_stem(func.__code__.co_filename)
    name_func = func.__name__
    log_file = f"{log_folder}/{relative_dir}/{name_file}/{name_func}.txt"
    path_mkdir(path_dirname(log_file), exist_ok=True, parents=True)
    return log_file
