"""One place that decides where BrowseGuard's logs go.

Every entry point calls configure() before doing anything else. Until this
module existed, logging happened to work because llm_client called
logging.basicConfig() as an import side effect — so it silently stopped the
moment browser_track stopped importing it. Configuration belongs to whoever
starts the process, not to whichever module happens to get imported first.

The tray app runs with no console attached, so a file handler isn't a nicety:
without it, an agent run that goes wrong on a user's machine leaves no trace
at all.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional, Union

LOG_PATH = Path(__file__).resolve().parent / "browseguard.log"

MAX_LOG_BYTES = 1024 * 1024
LOG_BACKUPS = 3

# threadName earns its place here: the agent answers each question on its own
# thread (see browser_track's "Assistant questions"), so without it two
# questions asked at once interleave into one unreadable trace.
LOG_FORMAT = "%(asctime)s [%(threadName)s] %(name)s %(levelname)s: %(message)s"

_configured = False


def configure(
    level: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = LOG_PATH,
) -> None:
    """Send logs to the console and to a rotating file.

    Safe to call more than once — later calls do nothing, so an entry point
    that imports another entry point doesn't end up double-logging every line.
    Set BROWSEGUARD_LOG_LEVEL=DEBUG to include message previews from the agent
    run; INFO reports the shape of a run without its contents.
    """
    global _configured
    if _configured:
        return
    _configured = True

    level = level or os.environ.get("BROWSEGUARD_LOG_LEVEL", "INFO").strip().upper()
    formatter = logging.Formatter(LOG_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    # Page titles and page text reach the console, and a Windows console is
    # cp1252, which cannot encode most of what the web contains. Without this,
    # one CJK or emoji page title turns a log line into a UnicodeEncodeError
    # that logging swallows — losing the line. Degrade to '?' instead.
    #
    # stdout gets the same treatment, and not only for symmetry: the entry
    # points print() the model's answer, which quotes those same page titles,
    # and there the error isn't swallowed — it ends the process mid-answer.
    for stream in (sys.stderr, sys.stdout):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError):
            # Not a real console (pythonw, a pipe, a captured stream).
            pass

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        try:
            rotating = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=MAX_LOG_BYTES,
                backupCount=LOG_BACKUPS,
                encoding="utf-8",
            )
            rotating.setFormatter(formatter)
            root.addHandler(rotating)
        except OSError:
            # A full or read-only disk shouldn't cost us console logging too.
            root.warning("Could not open %s; logging to the console only", log_file)

    # One INFO line per HTTP request to the model, saying nothing the agent
    # trace doesn't already say more legibly.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    root.info("Logging at %s (file: %s)", level, log_file)
