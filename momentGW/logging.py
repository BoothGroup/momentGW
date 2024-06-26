"""Logging."""

import contextlib
import os
import subprocess

import rich
from rich.console import Console
from rich.live import Live
from rich.panel import Panel  # noqa: F401
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
        "comment": "dim",
    }
)

console = Console(
    highlight=False,
    theme=theme,
)

silent = os.environ.get("MOMENTGW_SILENT", "0") == "1"

LIVE = None
STATUS = None
STATUS_MSGS = []
TABLE = None
LAYOUT = None
COMMENT = ""


def write(msg, *args, **kwargs):
    """Print a message to the console.

    Parameters
    ----------
    msg : str
        The message to print.
    args : tuple
        The arguments to format the message with.
    comment : str, optional
        A comment to print alongside the message.
    **kwargs : dict, optional
        Additional keyword arguments to pass to `console.print`.
    """

    # Check if we need to print the message
    if silent:
        return

    # Format the message
    if isinstance(msg, str) and args:
        msg = msg % args

    # See if the message has a comment
    if comment := (kwargs.pop("comment", None) or COMMENT):
        table = _Table.grid(expand=True)
        table.add_column("", justify="left")
        table.add_column("", justify="right", style="comment")
        table.add_row(msg, comment)
        msg = table

    # Print the message
    console.print(msg, **kwargs)


def warn(msg, *args, **kwargs):
    """Print a message to the console with a warning comment.

    Parameters
    ----------
    msg : str
        The message to print.
    args : tuple
        The arguments to format the message with.
    **kwargs : dict, optional
        Additional keyword arguments to pass to `console.print`.
    """

    # Add a warning comment
    kwargs["comment"] = "[bad]Warning![/]"

    # Print the message
    write(msg, *args, **kwargs)


def rate(value, good_threshold, ok_threshold, invert=False):
    """Return a colour rating based on a value and thresholds.

    Parameters
    ----------
    value : float
        The value to rate.
    good_threshold : float
        The threshold for a good rating.
    ok_threshold : float
        The threshold for an ok rating.
    invert : bool, optional
        Invert the rating. Default value is `False`.

    Returns
    -------
    style : str
        The style to use for the rating.
    """
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

    elif LIVE:
        # There is a live log and there is a status spinner and/or table
        LAYOUT = _Table.grid()
        if TABLE:
            LAYOUT.add_row(TABLE)
        if STATUS:
            LAYOUT.add_row(STATUS)
        LIVE.update(LAYOUT, refresh=False)


class Status:
    """A status spinner with nested status messages."""

    def __init__(self, msg, *args, **kwargs):
        self.msg = msg

    def __enter__(self):
        """Enter the context manager."""
        if not silent:
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
        if not silent:
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

    def add_column(self, *args, **kwargs):
        """Add a column to the table."""
        super().add_column(*args, **kwargs)
        if self._is_live and not silent:
            _update_live()

    add_column.__doc__ = _Table.add_column.__doc__

    def add_row(self, *args, **kwargs):
        """Add a row to the table."""
        super().add_row(*args, **kwargs)
        if self._is_live and not silent:
            _update_live()

    add_row.__doc__ = _Table.add_row.__doc__

    def __init__(self, *args, **kwargs):
        kwargs["show_edge"] = kwargs.get("show_edge", False)
        kwargs["show_header"] = kwargs.get("show_header", True)
        kwargs["expand"] = kwargs.get("expand", False)
        kwargs["title_style"] = kwargs.get("title_style", "bold")
        kwargs["header_style"] = kwargs.get("header_style", "")
        kwargs["box"] = kwargs.get("box", rich.box.SIMPLE)
        kwargs["padding"] = kwargs.get("padding", (0, 2))
        kwargs["collapse_padding"] = kwargs.get("collapse_padding", True)

        super().__init__(*args, **kwargs)

        self._is_live = False

    def __enter__(self):
        """Enter the context manager."""
        if not silent:
            global TABLE
            TABLE = self
            self._is_live = True
            _update_live()
        return self

    def __exit__(self, *args):
        """Exit the context manager."""
        if not silent:
            global TABLE
            TABLE = None
            self._is_live = False
            _update_live()


def time(msg, elapsed):
    """Record a time.

    Parameters
    ----------
    msg : str
        The message to record.
    elapsed : float
        The time elapsed.
    """
    if "_times" not in time.__dict__:
        time._times = {}
    time._times[msg] = time._times.get(msg, 0) + elapsed


def dump_times():
    """Print a table with the timings."""
    if "_times" in time.__dict__:
        table = Table(title="Timings")
        table.add_column("Task", justify="right")
        table.add_column("Time", justify="right")
        for msg, elapsed in time._times.items():
            table.add_row(msg, util.Timer.format_time(elapsed))
        write("")
        write(table)


def init_logging():
    """Initialise the logging with a header."""

    if globals().get("_MOMENTGW_LOG_INITIALISED", False):
        return

    # Print header
    header_size = max([len(line) for line in HEADER.split("\n")])
    space = " " * (header_size - len(__version__))
    write(f"[bold]{HEADER}[/bold]" % f"{space}[bold]{__version__}[/bold]")

    def get_git_hash(directory):
        """Get the git hash of a directory."""
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
    import h5py
    import numpy
    import pyscf
    import rich
    import scipy

    import momentGW

    packages = [numpy, scipy, h5py, pyscf, dyson, rich, momentGW]
    if mpi_helper.mpi is not None:
        import mpi4py

        packages.append(mpi4py)

    for module in packages:
        write(f"[bold]{module.__name__}:[/]")
        write(f" > Version:  {getattr(module, '__version__', 'N/A')}")
        write(
            " > Git hash: %s",
            get_git_hash(os.path.join(os.path.dirname(module.__file__), "..")),
        )

    # Environment variables
    threads = os.environ.get("OMP_NUM_THREADS", 1)
    write(f"[bold]OpenMP threads[/]: {threads if threads else 1}")
    write(f"[bold]MPI rank[/]: {mpi_helper.rank} of {mpi_helper.size}")

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
    # return Table(**kwargs)
    with Table(**kwargs) as table:
        yield table


@contextlib.contextmanager
def with_comment(comment):
    """Run a function with a comment."""
    global COMMENT
    COMMENT = comment
    yield
    COMMENT = ""


@contextlib.contextmanager
def with_silent():
    """Run a function silently."""
    global silent
    old_silent = silent
    silent = True
    yield
    silent = old_silent


@contextlib.contextmanager
def with_modifiers(**kwargs):
    """Run a function with modified logging."""
    functions = {
        "status": with_status,
        "timer": with_timer,
        "comment": with_comment,
    }
    with contextlib.ExitStack() as stack:
        for key, value in kwargs.items():
            stack.enter_context(functions[key](value))
        yield
