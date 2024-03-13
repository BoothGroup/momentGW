"""Logging."""

import contextlib
import os
import subprocess

import rich
from rich.console import Console
from rich.live import Live
from rich.status import Status as _Status
from rich.table import Table as _Table
from rich.theme import Theme

from momentGW import __version__, mpi_helper, util

HEADER = """                                       _    ______        __
  _ __ ___   ___  _ __ ___   ___ _ __ | |_ / ___\ \      / /
 | '_ ` _ \ / _ \| '_ ` _ \ / _ \ '_ \| __| |  _ \ \ /\ / /
 | | | | | | (_) | | | | | |  __/ | | | |_| |_| | \ V  V /
 |_| |_| |_|\___/|_| |_| |_|\___|_| |_|\__|\____|  \_/\_/
%s"""  # noqa: W605

theme = Theme(
    {
        "good": "green",
        "ok": "yellow",
        "bad": "red",
        "option": "bold cyan",
        "output": "bold blue",
        "comment": "italic dim",
    }
)

console = Console(
    highlight=False,
    theme=theme,
)

level = int(os.environ.get("MOMENTGW_LOG_LEVEL", "3"))

LIVE = None
STATUS = None
STATUS_MSGS = []
TABLE = None
LAYOUT = None
COMMENT = ""


def set_log_level(new_level):
    """Set the logging level."""
    global level
    level = new_level


def write(msg, *args, **kwargs):
    """Print a message to the console."""

    # Format the message
    if isinstance(msg, str) and args:
        msg = msg % args

    # See if the message has a comment
    if comment := (kwargs.pop("comment", None) or COMMENT):
        if isinstance(comment, str):
            space = console.width - len(msg) - len(comment)
            msg = f"{msg}{space * ' '}[comment]{comment}[/]"

    # Print the message
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


def rate(value, good_threshold, ok_threshold, invert=False):
    """Return a colour rating based on a value and thresholds."""
    if value < good_threshold:
        rating = "good" if not invert else "bad"
    elif value < ok_threshold:
        rating = "ok"
    else:
        rating = "bad" if not invert else "good"
    return rating


def _update_live():
    """Update the live log.

    Notes
    -----
    The live log can have a status spinner and/or an updating table.
    The `Live` object is created with `transient=True`, so it will be
    removed from the display after the context manager is exited.
    """

    global LIVE, LAYOUT

    if not LIVE and (STATUS or TABLE):
        # There is no live log, but there is a status spinner and/or
        # table waiting to be displayed
        LAYOUT = _Table.grid()
        if TABLE:
            LAYOUT.add_row(TABLE)
        if STATUS:
            LAYOUT.add_row(STATUS)
        LIVE = Live(LAYOUT, console=console, transient=True)
        LIVE.start()

    elif LIVE and not (STATUS or TABLE):
        # There is a live log, but there is no status spinner or table
        LIVE.refresh()
        LIVE.stop()
        LIVE = None
        console.clear_live()


class Status:
    """A status spinner with nested status messages."""

    def __init__(self, msg, *args, **kwargs):
        self.msg = msg

    def __enter__(self):
        """Enter the context manager."""
        if level >= 1:
            global LIVE, STATUS, STATUS_MSGS
            if STATUS is None:
                STATUS_MSGS = [self.msg]
                STATUS = _Status(self.msg, console=console)
            else:
                STATUS_MSGS.append(self.msg)
                STATUS.update(" > ".join(STATUS_MSGS))
            _update_live()
        return LIVE

    def __exit__(self, *args):
        """Exit the context manager."""
        if level >= 1:
            global LIVE, STATUS, STATUS_MSGS
            STATUS_MSGS = STATUS_MSGS[:-1]
            if not STATUS_MSGS:
                STATUS = None
            else:
                STATUS.update(" > ".join(STATUS_MSGS))
            _update_live()


class Table(_Table):
    """A table with additional context manager methods.

    Notes
    -----
    Since the `Live` object is created with `transient=True`, tables
    using the context manager will be removed from the display after
    the context manager is exited. Tables should be manually printed
    to the console if they are required to be displayed afterwards.
    """

    def __init__(self, *args, **kwargs):
        kwargs["show_edge"] = kwargs.get("show_edge", False)
        kwargs["show_header"] = kwargs.get("show_header", True)
        kwargs["expand"] = kwargs.get("expand", False)
        kwargs["title_style"] = kwargs.get("title_style", "bold")
        kwargs["header_style"] = kwargs.get("header_style", "")
        kwargs["box"] = kwargs.get("box", rich.box.SIMPLE)
        kwargs["padding"] = kwargs.get("padding", (0, 2))

        self.min_live_level = kwargs.pop("min_live_level", 1)

        super().__init__(*args, **kwargs)

    def __enter__(self):
        """Enter the context manager."""
        if level >= self.min_live_level:
            global LIVE, TABLE
            TABLE = self
            _update_live()
        return self

    def __exit__(self, *args):
        """Exit the context manager."""
        if level >= self.min_live_level:
            global LIVE, TABLE
            TABLE = None
            _update_live()

    def add_column(self, *args, **kwargs):
        """Add a column to the table."""
        super().add_column(*args, **kwargs)
        if level >= self.min_live_level and TABLE is self:
            _update_live()

    add_column.__doc__ = _Table.add_column.__doc__

    def add_row(self, *args, **kwargs):
        """Add a row to the table."""
        super().add_row(*args, **kwargs)
        if level >= self.min_live_level and TABLE is self:
            _update_live()

    add_row.__doc__ = _Table.add_row.__doc__


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


@contextlib.contextmanager
def with_timer(task_name):
    """Run a function with a timer."""
    timer = util.Timer()
    yield
    time(task_name, timer())


@contextlib.contextmanager
def with_status(task_name):
    """Run a function with a status spinner."""
    with Status(task_name):
        yield


@contextlib.contextmanager
def with_table(**kwargs):
    """Run a function with a table."""
    table = Table(**kwargs)
    yield table


@contextlib.contextmanager
def with_comment(comment):
    """Run a function with a comment."""
    global COMMENT
    COMMENT = comment
    yield
    COMMENT = ""


@contextlib.contextmanager
def with_log_level(new_level):
    """Run a function with a new log level."""
    old_level = level
    set_log_level(new_level)
    yield
    set_log_level(old_level)


@contextlib.contextmanager
def with_modifiers(**kwargs):
    """Run a function with modified logging."""
    functions = {
        "log_level": with_log_level,
        "status": with_status,
        "timer": with_timer,
        "comment": with_comment,
    }
    with contextlib.ExitStack() as stack:
        for key, value in kwargs.items():
            stack.enter_context(functions[key](value))
        yield
