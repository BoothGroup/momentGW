"""Logging."""

import functools
import os
import subprocess

import rich
from rich.console import Console
from rich.status import Status as _Status
from rich.table import Table

from momentGW import __version__, mpi_helper, util

HEADER = """                                       _    ______        __
  _ __ ___   ___  _ __ ___   ___ _ __ | |_ / ___\ \      / /
 | '_ ` _ \ / _ \| '_ ` _ \ / _ \ '_ \| __| |  _ \ \ /\ / /
 | | | | | | (_) | | | | | |  __/ | | | |_| |_| | \ V  V /
 |_| |_| |_|\___/|_| |_| |_|\___|_| |_|\__|\____|  \_/\_/
%s"""  # noqa: W605

console = Console(
    highlight=False,
)

Table = functools.partial(
    Table,
    show_edge=False,
    show_header=True,
    expand=False,
    title_style="bold",
    header_style="",
    box=rich.box.SIMPLE,
    padding=(0, 2),
)

level = int(os.environ.get("MOMENTGW_LOG_LEVEL", "3"))


def set_log_level(new_level):
    """Set the logging level."""
    global level
    level = new_level


def write(msg, *args, **kwargs):
    """Print a message to the console."""
    if isinstance(msg, str) and args:
        msg = msg % args
    console.print(msg, **kwargs)


def _write(msg, required_level, *args, **kwargs):
    """Print a message to the console if the level is high enough."""
    if level >= required_level:
        write(msg, *args, **kwargs)


def output(msg, *args, **kwargs):
    """Print an output message."""
    _write(msg, 1, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """Print a warning message."""
    _write(msg, 0, *args, **kwargs)


def error(msg, *args, **kwargs):
    """Print an error message."""
    _write(msg, 0, *args, **kwargs)


def info(msg, *args, **kwargs):
    """Print an info message."""
    _write(msg, 2, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """Print a debug message."""
    _write(msg, 3, *args, **kwargs)


class Status(_Status):
    """A status spinner with nested status messages."""

    _status = None
    _status_msgs = []

    def __init__(self, msg, *args, **kwargs):
        self.msg = msg

    def __enter__(self):
        """Enter the context manager."""
        if level >= 1:
            if self.__class__._status is None:
                self.__class__._status_msgs = [self.msg]
                self.__class__._status = console.status(self.msg)
            else:
                self.__class__._status_msgs.append(self.msg)
                self.__class__._status.update(" > ".join(self.__class__._status_msgs))
            self.__class__._status.__enter__()
            import time as _time

            _time.sleep(0.2)
        return self

    def __exit__(self, *args):
        """Exit the context manager."""
        if level >= 1:
            self.__class__._status_msgs = self.__class__._status_msgs[:-1]
            if not self.__class__._status_msgs:
                self.__class__._status.__exit__(*args)
                self.__class__._status = None
            else:
                self.__class__._status.update(" > ".join(self.__class__._status_msgs))


def time(msg, elapsed, *args, **kwargs):
    """Print a message with the time elapsed."""
    # if level >= 2:
    #    write(f"{msg} in {elapsed}", *args, **kwargs)
    if "_times" not in time.__dict__:
        time._times = {}
    time._times[msg] = time._times.get(msg, 0) + elapsed


def dump_times():
    """Print the times."""
    if "_times" in time.__dict__:
        table = Table(title="Timings")
        table.add_column("Task", justify="right")
        table.add_column("Time", justify="right")
        for msg, elapsed in time._times.items():
            table.add_row(msg, util.Timer.format_time(elapsed))
        debug("")
        output(table)


def init_logging():
    """Initialise the logging with a header."""

    if globals().get("_MOMENTGW_LOG_INITIALISED", False):
        return

    # Print header
    header_size = max([len(line) for line in HEADER.split("\n")])
    space = " " * (header_size - len(__version__))
    info(f"[bold]{HEADER}[/bold]" % f"{space}[bold]{__version__}[/bold]")

    # Print versions of dependencies and ebcc
    def get_git_hash(directory):
        git_directory = os.path.join(directory, ".git")
        cmd = ["git", "--git-dir=%s" % git_directory, "rev-parse", "--short", "HEAD"]
        try:
            git_hash = subprocess.check_output(
                cmd, universal_newlines=True, stderr=subprocess.STDOUT
            ).rstrip()
        except subprocess.CalledProcessError:
            git_hash = "N/A"
        return git_hash

    import dyson
    import numpy
    import pyscf

    import momentGW

    for module in (numpy, pyscf, dyson, momentGW):
        info(f"[bold]{module.__name__}:[/]")
        info(" > Version:  %s" % module.__version__)
        info(" > Git hash: %s" % get_git_hash(os.path.join(os.path.dirname(module.__file__), "..")))

    # Environment variables
    info("[bold]OMP_NUM_THREADS[/] = %s" % os.environ.get("OMP_NUM_THREADS", ""))
    info("[bold]MPI rank[/] = %d of %d" % (mpi_helper.rank, mpi_helper.size))

    globals()["_MOMENTGW_LOG_INITIALISED"] = True


def with_timer(task_name):
    """Run a function with a timer."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer = util.Timer()
            result = func(*args, **kwargs)
            time(task_name, timer())
            return result

        return wrapper

    return decorator


def with_status(task_name):
    """Run a function with a status spinner."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with Status(task_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
