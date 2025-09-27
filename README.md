# difflogtest

[![PyPI version](https://badge.fury.io/py/difflogtest.svg)](https://badge.fury.io/py/difflogtest)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unittest framework for reproducible testing across team environments. Ensures that all team members get identical outputs and logs, detecting environment inconsistencies or unintended code changes.

<p align="center">
  <img src="assets/DiffLogTest.svg" alt="difflogtest logo"/>
</p>

## Core Philosophy

I don't like writing tests. But in a team with multiple people, I need assurance that we're all running in the same environment and getting the same results. If outputs differ between team members, it indicates either environment setup issues or unwanted code changes. This library enforces reproducible testing.

## Features

- **Reproducible Testing**: Ensures identical outputs across all team environments
- **Output Comparison**: Automatically compares function outputs with logged expectations
- **Environment Validation**: Detects when environments differ between team members
- **Log Normalization**: Built-in normalization of timestamps, memory addresses, and other variable content
- **Flexible Matching**: Include/exclude patterns for files and functions
- **Parallel Execution**: CUDA-aware multiprocessing for performance
- **CLI Interface**: Easy command-line testing
- **Extensible**: Custom log replacement rules for any output format

## Installation

```bash
pip install difflogtest
# or with uv
uv add difflogtest
# then sync dependencies
uv sync
```

## Quick Start

### Basic Usage

First, create a logger and decorate your test functions with `@register_unittest(logger=logger)`:

```python
from difflogtest import register_unittest, get_logger

# Create a logger instance
logger = get_logger()

@register_unittest(logger=logger)
def test_my_function():
    """Test function that produces complex output."""
    logger.info("Starting complex computation...")
    logger.rule("Processing Data")

    result = my_complex_function()

    logger.success("Computation completed successfully")
    logger.info(f"Result shape: {result.shape}")

    return result  # This output will be compared with logged expectations

def my_complex_function():
    """Helper function that also logs.

    This function could be defined elsewhere, such as in another file.
    """
    logger.info("Inside complex computation...")
    logger.info("Processing step 1")
    # ... computation logic ...
    logger.info("Computation step completed")
    return {"result": 42, "shape": (10, 10)}
```

For functions that need special configuration during testing, pass additional kwargs:

```python
@register_unittest(logger=logger, confidence_threshold=0.1, model="special_model")
def test_special_function(confidence_threshold: float = 0.5, model: str = "default"):
    # Function parameters will use the unittest kwargs when called
    result = special_computation(confidence_threshold, model)
    return result
```

**How Logging Works:**
- All logger outputs (`logger.info()`, `logger.success()`, `logger.rule()`, etc.) are captured
- The decorated function's return value is also logged
- Everything gets written to a `.txt` file in the `unittest_logs_dir` directory
- File structure: `{unittest_logs_dir}/{relative_path}/{filename}/{function_name}.txt`

**Logger Sharing:**
- `get_logger()` uses `@lru_cache`, so all calls return the same logger instance
- Helper functions can call `get_logger()` and share the same logging context
- All logs from decorated functions and their helpers are captured together

Then run your unittests:

```python
from difflogtest import UnitTests

# Run all unittests - outputs go to unittest_logs/ by default
tests = UnitTests()

# Or specify custom log directory
tests = UnitTests(unittest_logs_dir="my_custom_logs")
```

### CLI Usage

After `uv sync`, the `run-unittests` command is available directly in your terminal:

```bash
# Run all unittests
run-unittests

# Run with substring patterns (wildcards added automatically)
run-unittests --include-file-pattern "test" --include-function-pattern "test_"

# Run specific function in specific file
run-unittests --include-file-pattern cli_image2camera --include-function-pattern scene_from_image

# Dry run to see what would be tested
run-unittests --dry-run

# Custom log directory
run-unittests --unittest-logs-dir "custom_logs"
```

Or using uv run:

```bash
uv run run-unittests --include-file-pattern "test"
```

## Quick Start

**Want to see difflogtest in action quickly? Here's the minimal setup:**

### Step 1: Create a test file

Create a file called `cli/test_example.py`:

```python
from difflogtest import register_unittest, get_logger

logger = get_logger()

@register_unittest(logger=logger)
def test_simple():
    return 42

@register_unittest(logger=logger)
def test_with_logging():
    logger.info("Running test...")
    result = compute_sum()  # Helper function also logs
    logger.success(f"Final result: {result}")
    return result

def compute_sum():
    """Helper function that shares the logger."""
    logger.info("Computing 2 + 2...")
    return 2 + 2

if __name__ == "__main__":
    # Example of running functions directly (normal execution)
    logger.info("Running directly:")
    result = test_simple()
    logger.info(f"test_simple() returned: {result}")

    result = test_with_logging()
    logger.info(f"test_with_logging() returned: {result}")

    logger.info("Or run tests with: run-unittests --include-file-pattern test_example")
```

### Step 2: Run the tests

```bash
uv sync
run-unittests --include-file-pattern test_example
```

### Step 3: Check the results

Look in the `unittest_logs/` directory - you'll see organized log files for each test function!

**That's it!** difflogtest will create baseline outputs on the first run, then compare future runs against them.

### Running from CLI vs. Unittest

**Direct Python execution** (`python cli/test_example.py`):
```bash
$ python cli/test_example.py
[INFO] Computing 2 + 2...
Test completed with result: 4
```
- **Normal output**: Logs go to console/terminal
- **No file capture**: Just standard program execution
- **Same performance**: No overhead from unittest framework

**Unittest execution** (`run-unittests --include-file-pattern cli/test_example`):
```bash
$ run-unittests --include-file-pattern cli/test_example
[16:31:11] [[INFO]] UnitTests initialized.                                                                                      core.py:126
[16:31:11] [[INFO]] CUDA deterministic: True                                                                                    core.py:127
[16:31:11] [[INFO]] Enable LRU cache: False                                                                                     core.py:128
[16:31:11] Found unittest: cli/test_example.py:test_simple at line 6 - cli/test_example.py:6                                    core.py:493
[16:31:11] Found unittest: cli/test_example.py:test_with_logging at line 10 - cli/test_example.py:10                            core.py:493
[16:31:11] [[INFO]] Total registered unittests found: 2                                                                         core.py:570
────────────────────────────────────── Running unittest: func='test_simple' from cli/test_example.py ──────────────────────────────────────
[16:31:13] [[✅SUCCESS]] UnitTest 1/2 - unittest_logs/cli/test_example/test_simple.txt passed. Elapsed time: 0.00m 1.86s        core.py:377
─────────────────────────────────── Running unittest: func='test_with_logging' from cli/test_example.py ───────────────────────────────────
[16:31:15] [[✅SUCCESS]] UnitTest 2/2 - unittest_logs/cli/test_example/test_with_logging.txt passed. Elapsed time: 0.00m 1.79s  core.py:377
────────────────────────────────────────────── UnitTests completed in 00d : 00h : 00m : 03s ───────────────────────────────────────────────
```

**Contents of `unittest_logs/.../cli/test_example/test_with_logging.txt`:**
```
# ----------------------------------------------------- #
# ------------------ UNITTEST LOG --------------------- #
# ----------------------------------------------------- #
[INFO] Running test...
[INFO] Computing 2 + 2...
[SUCCESS] Final result: 4

# ----------------------------------------------------- #
# ------------------ UNITTEST OUTPUT ------------------ #
# ----------------------------------------------------- #
4
```

**Minimal Overhead**: Just one `get_logger()` call!

## Core Concepts

### UnitTests Class

The `UnitTests` class is the heart of difflogtest. It discovers decorated functions, runs them in isolated processes, and compares their outputs.

#### Key Configuration Options

```python
from difflogtest import UnitTests

tests = UnitTests(
    # File and function patterns
    include_file_pattern=["*test*", "*spec*"],
    exclude_file_pattern=["*deprecated*"],
    include_function_pattern=["test_*", "spec_*"],
    exclude_function_pattern=["*skip*"],

    # Execution settings
    cuda_deterministic=True,      # Deterministic CUDA operations
    enable_lru_cache=False,       # Disable LRU caching during tests
    default_seed=42,              # Random seed for reproducibility
    timeout_minutes=20,           # Timeout per test (minutes)

    # Logging and output
    verbosity={"info": True, "debug": False, "warning": True, "error": True},
    unittest_logs_dir="my_test_logs",

    # Custom log processing (see below)
    log_replacements=[],  # Will be auto-filled with defaults
)
```

### How Logging & Output Capture Works

The `@register_unittest(logger)` decorator captures **everything** that gets logged:

1. **Logger Calls**: `logger.info()`, `logger.success()`, `logger.rule()`, `logger.error()`, etc.
2. **Function Return Value**: The final `return result` statement
3. **Exception Traces**: Any errors during execution

**Output Structure:**
```
# ----------------------------------------------------- #
# ------------------ UNITTEST LOG --------------------- #
# ----------------------------------------------------- #
[INFO] Starting expensive computation...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Processing Phase
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[SUCCESS] Computation completed!
[INFO] Final result: 42

# ----------------------------------------------------- #
# ------------------ UNITTEST OUTPUT ------------------ #
# ----------------------------------------------------- #
{'result': 42, 'status': 'success'}
```

**File Organization:**
```
unittest_logs_dir/
├── relative/path/to/
│   └── test_file.py/
│       ├── test_function_a.txt
│       └── test_function_b.txt
└── another/path/
    └── test_module.py/
        └── test_another_function.txt
```

**Pattern Matching:**
Patterns use substring matching - `"test"` becomes `"*test*"` internally, matching any file/function containing "test".

**Success/Failure Criteria:**
- **Success**: When no unittest log exists yet (creates baseline) OR when output matches the staged git file
- **Failure**: When output differs from staged git file OR when function throws an exception

### Log Replacement System

difflogtest automatically normalizes variable content in logs and outputs. The system is fully customizable:

```python
from difflogtest import LogReplacement, add_log_replacement, UnitTests

# Get default replacements
replacements = LogReplacement.create_defaults()

# Add custom replacement
custom = LogReplacement(
    name="api_key",
    pattern=r"api_key_[a-zA-Z0-9]+",
    replacement="<API_KEY>",
    use_regex=True
)
add_log_replacement(replacements, custom)

# Use with UnitTests
tests = UnitTests()
tests.log_replacements = replacements
```

#### Built-in Replacements

difflogtest includes these default replacements:

- **Timestamps**: `2025-01-15 14:30:22` → `<TIMESTAMP>`
- **Memory Addresses**: `0x7f8b8c0d5e10` → `<MEMORY_ADDRESS>`
- **Home Directories**: `/home/user/...` → `<HOME_DIR>`
- **Git Hashes**: `a1b2c3d4e5f6...` → `<COMMIT_HASH>`
- **UUIDs**: `12345678-1234-1234-1234-123456789abc` → `<UUID_HASH>`
- **HuggingFace Snapshots**: `snapshots/abcdef123456` → `<HUGGINGFACE_SNAPSHOT>`
- **Temporary Files**: `/tmp/tmp123` → `<TEMPORAL_FILE>`

## Detailed Examples

### Example 1: Basic Function Testing

```python
import numpy as np
from difflogtest import register_unittest, get_logger

logger = get_logger()

@register_unittest(logger=logger)
def test_matrix_operations():
    """Test matrix multiplication and normalization."""
    # Generate test matrices
    A = np.random.randn(100, 50)
    B = np.random.randn(50, 200)

    # Perform computation
    result = np.dot(A, B)
    normalized = result / np.linalg.norm(result)

    return {
        "shape": result.shape,
        "mean": float(np.mean(normalized)),
        "std": float(np.std(normalized)),
        "frobenius_norm": float(np.linalg.norm(result, 'fro'))
    }
```

### Example 2: Image Processing Test

```python
import cv2
import numpy as np
from difflogtest import register_unittest, get_logger

logger = get_logger()

@register_unittest(logger=logger)
def test_image_filtering():
    """Test image filtering pipeline."""
    # Load test image
    img = cv2.imread("test_image.jpg")
    if img is None:
        raise ValueError("Test image not found")

    # Apply filters
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)

    # Calculate statistics
    stats = {
        "original_shape": img.shape,
        "gray_mean": float(np.mean(gray)),
        "blur_std": float(np.std(blurred)),
        "edge_count": int(np.sum(edges > 0)),
        "edge_percentage": float(np.sum(edges > 0) / edges.size * 100)
    }

    return stats
```

### Example 3: Machine Learning Model Test

```python
import torch
import torch.nn as nn
from difflogtest import register_unittest, get_logger

logger = get_logger()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

@register_unittest(logger=logger)
def test_model_inference():
    """Test ML model inference consistency."""
    # Set deterministic behavior
    torch.manual_seed(42)

    # Create model and test input
    model = SimpleModel()
    test_input = torch.randn(32, 10)

    # Run inference
    with torch.no_grad():
        output = model(test_input)

    # Return comprehensive results
    return {
        "output_shape": list(output.shape),
        "output_mean": float(torch.mean(output).item()),
        "output_std": float(torch.std(output).item()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(output.device),
        "dtype": str(output.dtype)
    }
```

### Example 4: Custom Log Replacements

```python
from difflogtest import UnitTests, LogReplacement, add_log_replacement

# Create custom replacements for your domain
custom_replacements = LogReplacement.create_defaults()

# Add domain-specific replacements
replacements_to_add = [
    LogReplacement(
        name="user_id",
        pattern=r"user_id_\d+",
        replacement="<USER_ID>",
        use_regex=True
    ),
    LogReplacement(
        name="session_token",
        pattern=r"session_[a-f0-9]{32}",
        replacement="<SESSION_TOKEN>",
        use_regex=True
    ),
    LogReplacement(
        name="database_url",
        pattern=r"postgresql://[^@\s]+@[^/\s]+/\w+",
        replacement="<DATABASE_URL>",
        use_regex=True
    ),
    # Literal string replacement (no regex)
    LogReplacement(
        name="version",
        pattern="v1.2.3",
        replacement="<VERSION>",
        use_regex=False
    )
]

for replacement in replacements_to_add:
    add_log_replacement(custom_replacements, replacement)

# Use with UnitTests
tests = UnitTests()
tests.log_replacements = custom_replacements
```

## Advanced Configuration

### Custom Test Discovery

```python
tests = UnitTests(
    # Only test specific files
    include_file_pattern=["*ml*", "*ai*"],
    exclude_file_pattern=["*deprecated*", "*experimental*"],

    # Only test specific functions
    include_function_pattern=["test_*", "validate_*"],
    exclude_function_pattern=["*slow*", "*flaky*"],

    # Custom output directory
    unittest_logs_dir="my_custom_test_logs"
)
```

### Performance Tuning

```python
tests = UnitTests(
    # CUDA settings for GPU tests
    cuda_deterministic=True,  # Ensure reproducible GPU operations
    enable_lru_cache=False,   # Disable caching during tests

    # Timeout and seeding
    timeout_minutes=30,       # 30-minute timeout per test
    default_seed=12345,       # Custom seed for reproducibility

    # Logging verbosity
    verbosity={
        "info": True,
        "debug": False,    # Reduce noise
        "warning": True,
        "error": True,
        "success": True
    }
)
```

### CLI Advanced Usage

```bash
# Run with substring patterns (wildcards added automatically)
run-unittests \
  --include-file-pattern "test" \
  --include-function-pattern "test_" \
  --exclude-function-pattern "slow"

# Run specific function in specific file
run-unittests \
  --include-file-pattern cli_image2camera \
  --include-function-pattern scene_from_image

# Run in dry-run mode to preview
run-unittests --dry-run

# Custom log directory
run-unittests --unittest-logs-dir "custom_logs"
```

## API Reference

### UnitTests

Main testing class with the following parameters:

- `skip_dirs`: Directories to skip during discovery
- `include_file_pattern`: File patterns to include (substring match, wildcards added automatically)
- `exclude_file_pattern`: File patterns to exclude (substring match, wildcards added automatically)
- `include_function_pattern`: Function patterns to include (substring match, wildcards added automatically)
- `exclude_function_pattern`: Function patterns to exclude (substring match, wildcards added automatically)
- `dry_run`: Preview mode without running tests
- `enable_lru_cache`: Enable/disable LRU caching
- `cuda_deterministic`: Enable deterministic CUDA operations
- `default_seed`: Random seed for reproducibility
- `timeout_minutes`: Timeout per test in minutes
- `verbosity`: Logging verbosity configuration
- `unittest_logs_dir`: Directory for test logs (default: "unittest_logs")
- `log_replacements`: Custom log processing rules

### LogReplacement

Configuration for log content replacement:

- `name`: Identifier for the replacement rule
- `pattern`: Regex pattern or literal string to match
- `replacement`: String to replace matches with
- `use_regex`: Whether to use regex (True) or literal matching (False)

### register_unittest

Decorator for registering test functions:

```python
register_unittest(logger=logger, **default_kwargs)
```

- `logger`: Logger instance - all `logger.info()`, `logger.success()`, `logger.rule()`, etc. calls within the decorated function are captured to log files
- `**default_kwargs`: Default keyword arguments for the decorated function - these override the function's default parameters during testing

**Note**: The logger parameter must be passed as a keyword argument (`logger=logger`). All logging output from the decorated function is automatically captured and written to timestamped log files. Additional kwargs allow testing functions with special configurations.

### Utility Functions

- `LogReplacement.create_defaults()`: Get default replacement rules
- `add_log_replacement(replacements, replacement)`: Add custom replacement
- `remove_log_replacement(replacements, name)`: Remove replacement by name
- `clear_log_replacements(replacements)`: Clear all replacements
- `reset_log_replacements(replacements)`: Reset to defaults
- `process_log_content(content, replacements)`: Process content with replacements

## Code Examples

### Function Registration with Logging

```python
from difflogtest import register_unittest, get_logger

logger = get_logger()

@register_unittest(logger=logger)
def my_test_function():
    logger.info("Starting expensive computation...")
    logger.rule("Processing Phase")

    result = expensive_computation()

    logger.success("Computation completed!")
    logger.info(f"Final result: {result}")

    return result  # All logger output + return value → unittest_logs_dir

def expensive_computation():
    """Helper function that shares the same logger in some other file."""
    logger.info("Running complex algorithm...")
    logger.info("Step 1: Data preparation")
    logger.info("Step 2: Computation")
    logger.info("Algorithm completed")
    return 42
```

*Logger sharing via LRU cache allows helper functions to contribute to the same log output*

### Custom Test Configuration

```python
from difflogtest import UnitTests

tests = UnitTests(
    include_file_pattern=["*test*"],
    cuda_deterministic=True,
    timeout_minutes=15,
    default_seed=42,
    verbosity={
        "info": True,
        "debug": False,
        "warning": True,
        "error": True
    }
)
```

*Comprehensive configuration with all major options*

### Log Processing Customization

```python
from difflogtest import LogReplacement, add_log_replacement

# Start with defaults
replacements = LogReplacement.create_defaults()

# Add domain-specific replacements
custom = LogReplacement(
    name="secret_key",
    pattern=r"sk-[a-zA-Z0-9]{48}",
    replacement="<SECRET_KEY>",
    use_regex=True
)
add_log_replacement(replacements, custom)
```

*Building on defaults with custom replacements*

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## License

MIT License - see LICENSE file for details.

---

For teams that need reproducible testing across different environments.
