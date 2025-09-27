"""Core unittest functionality for difflogtest."""

import ast
import asyncio
import concurrent.futures
import importlib.util
import io
import multiprocessing as mp
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from dataclasses import field
from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from .logging import DEFAULT_VERBOSITY, LoggingRich, seed_everything
from .utils import (
    LogReplacement,
    disable_debug_mode,
    enable_debug_mode,
    get_elapsed_time,
    process_log_content,
    set_unittest_mode,
)
from .utils.path import (
    is_file_changed,
    path_absolute,
    path_cwd,
    path_dirname,
    path_from_pattern,
    path_is_dir,
    path_join,
    path_listdir,
    path_open,
    path_relative_to,
    path_stem,
)

if TYPE_CHECKING:
    from importlib.machinery import ModuleSpec


SKIPPED_DIRECTORIES = [
    "dependencies",
    "scripts",
    "data",
    "pretrained_models",
    "logs",
]


@dataclass(config=ConfigDict(extra="forbid"))
class UnitTests:
    """Unit tests class."""

    skip_dirs: list[str] = field(default_factory=lambda: SKIPPED_DIRECTORIES)
    """ The directories to skip. """

    include_file_pattern: list[str] = field(default_factory=lambda: ["*"])
    """ The file patterns to include. """

    exclude_file_pattern: list[str] = field(default_factory=lambda: ["NULL"])
    """ The file patterns to exclude. """

    include_function_pattern: list[str] = field(default_factory=lambda: ["*"])
    """ The function patterns to include. """

    exclude_function_pattern: list[str] = field(
        default_factory=lambda: ["NULL"],
    )
    """ The function patterns to exclude. """

    dry_run: bool = False
    """ If True, the unittest will be returned as a list of strings. """

    enable_lru_cache: bool = False
    """ If True, the unittest will enable LRU cache. """

    cuda_deterministic: bool = True
    """ If True, the unittest will use deterministic CUDA operations. """

    default_seed: int = 42
    """ Default seed value for random number generation. """

    timeout_minutes: int = 20
    """ Timeout in minutes for individual test execution. """

    unittest_logs_dir: str = "unittest_logs"
    """ Directory to store the unittest logs. """

    verbosity: dict[str, bool] = field(
        default_factory=lambda: DEFAULT_VERBOSITY
    )
    """ Logging verbosity configuration. """

    log_replacements: list[LogReplacement] = field(default_factory=list)
    r""" List of log replacement rules for consistent output.

    Modify this list to customize log processing. Use LogReplacement.create_defaults()
    to get default rules, then add/remove as needed:

    Example:
        >>> from difflogtest import LogReplacement, add_log_replacement
        >>> tests = UnitTests()
        >>> custom = LogReplacement("my_pattern", r"secret_\w+", "<SECRET>")
        >>> add_log_replacement(tests.log_replacements, custom)
    """

    logger: LoggingRich = field(init=False)
    """ The logger for the UnitTests instance. """

    def __post_init__(self) -> None:
        """Execute a series of operations after the initialization of a UnitTests instance."""
        # Initialize logger with instance verbosity
        self.logger = LoggingRich(verbosity=self.verbosity)

        # Set up default log replacements if none provided
        if not self.log_replacements:
            self.log_replacements = LogReplacement.create_defaults()

        self.check_inputs()
        self.logger.info("UnitTests initialized.")
        self.logger.info(f"CUDA deterministic: {self.cuda_deterministic}")
        self.logger.info(f"Enable LRU cache: {self.enable_lru_cache}")
        set_unittest_mode(
            disable_lru_cache=not self.enable_lru_cache,
            cuda_deterministic=self.cuda_deterministic,
        )
        time_start = time.time()
        self.process()
        elapsed_time = time.time() - time_start
        self.logger.rule(
            f"UnitTests completed in {get_elapsed_time(elapsed_time)}",
        )

    def check_inputs(self) -> None:
        """Check if the provided input values are valid."""
        is_exit = False
        if not any(self.include_file_pattern):
            self.logger.print_exception(
                "No valid include file patterns found."
            )
            is_exit = True
        if not any(self.exclude_file_pattern):
            self.logger.print_exception(
                "No valid exclude file patterns found."
            )
            is_exit = True
        if not any(self.include_function_pattern):
            self.logger.print_exception(
                "No valid include function patterns found.",
            )
            is_exit = True
        if not any(self.exclude_function_pattern):
            self.logger.print_exception(
                "No valid exclude function patterns found.",
            )
            is_exit = True

        # if "*" is the only pattern, it will match everything, and it should only be length 1
        if (
            "*" in self.include_file_pattern
            and len(self.include_file_pattern) > 1
        ):
            self.logger.print_exception(
                "The include_file_pattern should only contain '*' to match all files.",
            )
            is_exit = True
        if (
            "*" in self.include_function_pattern
            and len(self.include_function_pattern) > 1
        ):
            self.logger.print_exception(
                "The include_function_pattern should only contain '*' to match all functions.",
            )
            is_exit = True
        # "*" cant be in the exclude patterns and include patterns at the same time
        if (
            "*" in self.exclude_file_pattern
            and "*" in self.include_file_pattern
        ):
            self.logger.print_exception(
                "The include_file_pattern and exclude_file_pattern cannot both contain '*'.",
            )
            is_exit = True
        if (
            "*" in self.exclude_function_pattern
            and "*" in self.include_function_pattern
        ):
            self.logger.print_exception(
                "The include_function_pattern and exclude_function_pattern cannot both contain '*'.",
            )
            is_exit = True

        if is_exit:
            sys.exit(1)

    @staticmethod
    def get_decorated_functions(
        filename: str, logger_sys: LoggingRich
    ) -> tuple[list[str], list[int]]:
        """Extract names of functions decorated with @register_unittest from a Python file."""
        try:
            with path_open(filename, "r", encoding="utf-8") as file:
                tree = ast.parse(file.read(), filename=filename)

            functions = []
            line_number = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    for decorator in node.decorator_list:
                        if (
                            isinstance(decorator, ast.Call)
                            and isinstance(decorator.func, ast.Name)
                            and decorator.func.id == "register_unittest"
                        ):
                            functions.append(node.name)
                            line_number.append(node.lineno)
                            logger_sys.debug(
                                f"Found unittest: {node.name} at line {node.lineno} in {filename}"
                            )
                        else:
                            continue
        except Exception:
            logger_sys.print_exception(f"Error parsing {filename}")

        # sort the functions and line numbers, line numbers based on the order of the functions
        index = sorted(range(len(functions)), key=lambda x: functions[x])
        functions = [functions[i] for i in index]
        line_number = [line_number[i] for i in index]

        return functions, line_number

    @staticmethod
    def run_func(module_name: str, func_name: str) -> str:
        """Run a function in a subprocess."""
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        with redirect_stdout(io.StringIO()):
            if asyncio.iscoroutinefunction(func):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(func())
                finally:
                    loop.close()
            else:
                result = func()
        return str(result)

    def execute_function_in_subprocess(
        self,
        module_name: str,
        func_name: str,
    ) -> str:
        """Execute a function and clean up after."""
        # spawn is required for CUDA operations
        # if multiprocessing.get_start_method(allow_none=True) != "spawn":
        #     multiprocessing.set_start_method("spawn", force=True)

        # Use ProcessPoolExecutor instead of ThreadPoolExecutor for CUDA operations
        seed_everything(
            self.default_seed,
            verbose=False,
            cuda_deterministic=self.cuda_deterministic,
        )
        with concurrent.futures.ProcessPoolExecutor(
            mp_context=mp.get_context("spawn"), initializer=self.init_cuda
        ) as executor:
            try:
                future = executor.submit(
                    UnitTests.run_func, module_name, func_name
                )
                return future.result(
                    timeout=self.timeout_minutes * 60
                )  # Configurable timeout
            except concurrent.futures.TimeoutError:
                traceback_error = traceback.format_exc()
                return f"Error during execution: Timeout\n\n{traceback_error}"
            except Exception as e:
                traceback_error = traceback.format_exc()
                return f"Error during execution: {e}\n\n{traceback_error}"

    @staticmethod
    def init_cuda() -> None:
        """Initialize CUDA if available."""
        try:
            from tools.pytorch.torch_tools import init_cuda

            init_cuda()
        except ImportError:
            pass

    def worker(self, module_name: str, func_name: str) -> tuple[str, bool]:
        """Execute a given command inside a subprocess."""
        enable_debug_mode()
        result = self.execute_function_in_subprocess(
            module_name,
            func_name,
        )
        success = "error during execution" not in result.lower()
        self.release_cuda_memory()
        disable_debug_mode()
        return result, success

    @staticmethod
    def release_cuda_memory() -> None:
        """Release CUDA memory if available."""
        try:
            from tools.pytorch.torch_tools import release_cuda_memory

            release_cuda_memory()
        except ImportError:
            pass

    def call_function_and_get_result(
        self,
        module: Any,
        func_name: str,
    ) -> tuple[str, bool, float]:
        """Execute a specific function within a given module and measure its execution time."""
        # Start timing
        time_start = time.time()

        # Retrieve module name from the module object
        import_module_name = path_relative_to(
            path_absolute(module.__file__),
            path_cwd(),
        )
        module_name = (
            str(import_module_name).replace("/", ".").replace(".py", "")
        )
        time_start = time.time()

        # Create and start the subprocess
        result, success = self.worker(module_name, func_name)

        time_end = time.time()
        return str(result), success, time_end - time_start

    @staticmethod
    def diff_and_write(
        file_path: str,
        content: str,
        id_number: int,
        total_units: int,
        elapsed_time: float,
        *,
        success: bool,
        logger_sys: LoggingRich,
    ) -> bool:
        """Compare new content with existing file content and writes the new content if different."""
        write_append = "w" if success else "a"
        with path_open(file_path, write_append) as file:
            file.write(content)

        elapsed_time_min_sec = (
            f"{elapsed_time // 60:.2f}m {elapsed_time % 60:.2f}s"
        )

        if not success or is_file_changed(file_path):
            logger_sys.error(
                f"UnitTest {id_number}/{total_units} FAILED! - Content mismatch in [bold italic]{file_path}[/bold italic] or there was an error. "
                "Hint: First make sure you can run this function without errors.\n"
                "Once it runs without errors, run unittest again with the command:\n"
                "\t[italic dark_goldenrod]run-unittests "
                f"--include-file-pattern {path_stem(path_dirname(file_path))} "
                f"--include-function-pattern {path_stem(file_path)}[/italic dark_goldenrod]",
            )
            logger_sys.error(f"Elapsed time: {elapsed_time_min_sec}")
            unittests_success = False
        else:
            logger_sys.success(
                f"UnitTest {id_number}/{total_units} - {file_path} passed. Elapsed time: {elapsed_time_min_sec}",
            )
            unittests_success = True
        return unittests_success

    @property
    def root_dirs(self) -> list[str]:
        """Get all the directories in the current project."""
        # Get current working directory
        cwd = path_cwd()
        # Get all directories in cwd that aren't in skip_dirs
        root_dirs = []
        dir_names = [
            i
            for i in path_listdir(
                cwd, include_hidden=False, include_private=False
            )
            if path_is_dir(i)
        ]
        for dir_name in dir_names:
            if any(pattern in dir_name for pattern in self.skip_dirs):
                continue
            root_dirs.append(dir_name)
        self.logger.debug(f"Root directories: {root_dirs}")
        return root_dirs

    def check_all_unittests(self) -> None:
        """Get all the unit tests in specified root directories."""
        paths: list[str] = []
        for root_dir in sorted(self.root_dirs):
            for subdir, _, files in os.walk(root_dir):
                for file in sorted(files):
                    if file.endswith(".py"):
                        file_path = path_join(subdir, file)
                        self.logger.debug(
                            f"Looking for unittests in {file_path}"
                        )
                        functions, _ = UnitTests.get_decorated_functions(
                            file_path,
                            self.logger,
                        )
                        for func in sorted(functions):
                            file_path_line = f"{file_path}:{func}"
                            paths.append(file_path_line)

        paths = sorted(paths)
        current_txt = []
        for path in paths:
            _path, _func = path.split(":")
            current_txt.append(self.get_unittest_file(_path, _func))

        # get all txt files
        txt_files = path_from_pattern(
            self.unittest_logs_dir,
            pattern=[".txt"],
            recursive=True,
            output_type="str",
        )
        error_files = []
        for saved_txt in txt_files:
            if saved_txt not in current_txt:
                error_files.append(saved_txt)

        if error_files:
            _files = "\n".join([f"*-> {_file} <-*" for _file in error_files])
            self.logger.error(
                f"[magenta]{_files}[/magenta]\nUnittest is not registered. This means it was removed from the codebase. "
                f"If so, please remove the file from the {self.unittest_logs_dir} folder."
            )
            sys.exit(1)

    def count_total_unittests(self) -> int:
        """Count the total number of unit tests in specified root directories."""
        total_tests = 0
        paths = []
        for root_dir in self.root_dirs:
            for subdir, _, files in os.walk(root_dir):
                for file in sorted(files):
                    if file.endswith(".py"):
                        file_path = path_join(subdir, file)
                        if not self.include_file(file_path):
                            self.logger.debug(
                                f"Skipping {file_path} due to pattern",
                            )
                            continue
                        self.logger.debug(f"Processing {file_path}")
                        functions, line_number = (
                            UnitTests.get_decorated_functions(
                                file_path,
                                self.logger,
                            )
                        )
                        number_of_functions = 0
                        for func, lineno in zip(
                            functions,
                            line_number,
                            strict=False,
                        ):
                            if self.include_function(func):
                                number_of_functions += 1
                                file_path_line = f"{file_path}:{func} at line {lineno} - {file_path}:{lineno}"
                                paths.append(file_path_line)
                            else:
                                self.logger.debug(
                                    f"Skipping {func} from {file_path} due to pattern",
                                )
                        total_tests += number_of_functions

        if not total_tests:
            self.logger.print_exception(
                "No unittests found. Please check the decorators.",
            )
            sys.exit(1)

        for path in sorted(paths):
            self.logger.log(
                f"Found unittest: [italic gold1]{path}[/italic gold1]"
            )

        return total_tests

    def include_file(self, file_path: str) -> bool:
        """Determine whether a file should be included based on provided include and exclude patterns."""
        # pattern can contain * to match all files
        if (
            len(self.exclude_file_pattern) != 1
            or self.exclude_file_pattern[0] != "NULL"
        ) and any(
            pattern in file_path for pattern in self.exclude_file_pattern
        ):
            return False
        return bool(
            (
                len(self.include_file_pattern) == 1
                and "*" in self.include_file_pattern
            )
            or any(
                pattern in file_path for pattern in self.include_file_pattern
            )
        )

    def include_function(self, func_name: str) -> bool:
        """Determine whether a function should be included based on given patterns."""
        # pattern can contain * to match all functions
        if (
            len(self.exclude_function_pattern) != 1
            or self.exclude_function_pattern[0] != "NULL"
        ) and any(
            pattern in func_name for pattern in self.exclude_function_pattern
        ):
            return False
        return bool(
            (
                len(self.include_function_pattern) == 1
                and "*" in self.include_function_pattern
            )
            or any(
                pattern in func_name
                for pattern in self.include_function_pattern
            )
        )

    def get_unittest_file(self, file_path: str, func: str) -> str:
        """Get the unittest file path for a given Python file."""
        relative_dir = path_relative_to(
            path_dirname(file_path),
            path_cwd(),
        )
        name_file = path_stem(file_path)
        return (
            f"{self.unittest_logs_dir}/{relative_dir}/{name_file}/{func}.txt"
        )

    @staticmethod
    def get_lineno(file_path: str, func: str) -> int:
        """Get the line number of a function in a file."""
        with path_open(file_path, "r", encoding="utf-8") as file:
            tree = ast.parse(file.read(), filename=file_path)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name == func
            ):
                return node.lineno
        msg = f"Function {func} not found in {file_path}"
        raise ValueError(msg)

    def process(self) -> None:
        """Process and run the unit tests."""
        self.check_all_unittests()
        total_units = self.count_total_unittests()
        self.logger.info(f"Total registered unittests found: {total_units}")
        if self.dry_run:
            sys.exit()
        if total_units == 0:
            self.logger.print_exception(
                "No unittests found. Please check the decorators.",
            )
            sys.exit(1)
        count = 1
        for root_dir in sorted(self.root_dirs):
            for subdir, _, files in os.walk(root_dir):
                for file in sorted(files):
                    if file.endswith(".py"):
                        file_path = path_join(subdir, file)
                        if not self.include_file(file_path):
                            continue
                        functions = UnitTests.get_decorated_functions(
                            file_path,
                            self.logger,
                        )[0]
                        if not functions:
                            continue
                        module_name = os.path.splitext(file)[0]  # noqa: PTH122
                        # Create a module specification from the file location
                        # This allows dynamic loading of Python modules from files
                        # module_name: Name of the module to create
                        # file_path: Path to the Python file to load
                        spec: ModuleSpec | None = (
                            importlib.util.spec_from_file_location(
                                module_name,
                                file_path,
                            )
                        )

                        if spec is None:
                            self.logger.print_exception(
                                f"Failed to create module spec for {file_path}",
                            )
                            sys.exit(1)

                        # Create a new module object from the specification
                        # This creates a blank module that will be populated with the code
                        module = importlib.util.module_from_spec(spec)

                        # Add the module to sys.modules to ensure proper module tracking
                        # Reference: https://github.com/mkdocs/mkdocs/issues/3141
                        sys.modules[module_name] = module

                        # Execute the module's code to initialize it
                        # This runs the actual Python code in the module, making its contents available
                        # The type ignore is needed because loader could be None, but we know it exists
                        # spec.loader.exec_module(module)  # type: ignore[union-attr]

                        for func in sorted(functions):
                            if not self.include_function(func):
                                continue
                            try:
                                self.logger.rule(
                                    f"Running unittest: {func=} from {file_path}",
                                )
                                result, success, elapsed_time = (
                                    self.call_function_and_get_result(
                                        module,
                                        func,
                                    )
                                )

                                # Process log content with configured replacements
                                processed_result = process_log_content(
                                    result, self.log_replacements
                                )

                                output_file = self.get_unittest_file(
                                    str(module.__file__), func
                                )
                                unittests_success = UnitTests.diff_and_write(
                                    output_file,
                                    processed_result,
                                    id_number=count,
                                    total_units=total_units,
                                    elapsed_time=elapsed_time,
                                    success=success,
                                    logger_sys=self.logger,
                                )
                                if not unittests_success:
                                    lineno = UnitTests.get_lineno(
                                        file_path, func
                                    )
                                    self.logger.error(
                                        f"Error in {file_path}:{lineno}",
                                    )
                                count += 1
                            except Exception:  # noqa: S110
                                pass
                                # logger_sys.print_exception(
                                #     f"Error in {func} from {file_path}: {e}",
                                # )
