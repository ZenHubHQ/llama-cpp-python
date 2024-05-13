import os
import sys
import psutil
import subprocess

from typing import Any, Dict, List, Tuple, Union

# Avoid "LookupError: unknown encoding: ascii" when open() called in a destructor
outnull_file = open(os.devnull, "w")
errnull_file = open(os.devnull, "w")

STDOUT_FILENO = 1
STDERR_FILENO = 2

class suppress_stdout_stderr(object):
    # NOTE: these must be "saved" here to avoid exceptions when using
    #       this context manager inside of a __del__ method
    sys = sys
    os = os

    def __init__(self, disable: bool = True):
        self.disable = disable

    # Oddly enough this works better than the contextlib version
    def __enter__(self):
        if self.disable:
            return self

        self.old_stdout_fileno_undup = STDOUT_FILENO
        self.old_stderr_fileno_undup = STDERR_FILENO

        self.old_stdout_fileno = self.os.dup(self.old_stdout_fileno_undup)
        self.old_stderr_fileno = self.os.dup(self.old_stderr_fileno_undup)

        self.old_stdout = self.sys.stdout
        self.old_stderr = self.sys.stderr

        self.os.dup2(outnull_file.fileno(), self.old_stdout_fileno_undup)
        self.os.dup2(errnull_file.fileno(), self.old_stderr_fileno_undup)

        self.sys.stdout = outnull_file
        self.sys.stderr = errnull_file
        return self

    def __exit__(self, *_):
        if self.disable:
            return

        # Check if sys.stdout and sys.stderr have fileno method
        self.sys.stdout = self.old_stdout
        self.sys.stderr = self.old_stderr

        self.os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        self.os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        self.os.close(self.old_stdout_fileno)
        self.os.close(self.old_stderr_fileno)


class MetaSingleton(type):
    """
    Metaclass for implementing the Singleton pattern.
    """

    _instances: Dict[type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Singleton(object, metaclass=MetaSingleton):
    """
    Base class for implementing the Singleton pattern.
    """

    def __init__(self):
        super(Singleton, self).__init__()


# Get snapshot of RAM and GPU usage before and after function execution.
# Adapted from: https://github.com/abetlen/llama-cpp-python/issues/223#issuecomment-1556203616
def get_cpu_usage(pid) -> float:
    """
    CPU usage in percentage by the current process.
    """
    process = psutil.Process(pid)
    return process.cpu_percent()

def get_ram_usage(pid) -> float:
    """
    RAM usage in MiB by the current process.
    """
    process = psutil.Process(pid)
    ram_info = process.memory_info()
    ram_usage = ram_info.rss / (1024 * 1024)  # Convert to MiB
    return ram_usage

def get_gpu_info_by_pid(pid) -> float:
    """
    GPU memory usage by the current process (if GPU is available)
    """
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"]).decode("utf-8")
        gpu_info = gpu_info.strip().split("\n")
        for info in gpu_info:
            gpu_pid, gpu_ram_usage = info.split(", ")
            if int(gpu_pid) == pid:
                return float(gpu_ram_usage.split()[0])
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return 0.0

def get_gpu_general_info() -> Tuple[float, float, float]:
    """
    GPU general info (if GPU is available)
    """
    try:
        gpu_info = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free", "--format=csv,noheader"]).decode("utf-8")
        gpu_utilization, gpu_memory_used, gpu_memory_free = gpu_info.strip().split("\n")[0].split(", ")
        return tuple(float(tup.split()[0]) for tup in [gpu_utilization, gpu_memory_used, gpu_memory_free])
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return 0.0, 0.0, 0.0
