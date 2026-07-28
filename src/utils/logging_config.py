"""Centralized logging configuration for the project.

Library modules (e.g. ``src.data.brats_reader``) only ever call
``logging.getLogger(__name__)`` and never configure handlers themselves -
this is standard Python logging practice, and keeps the library usable both
as a script and when imported into a notebook or app. Call
:func:`setup_logging` once, from a script/notebook/app entry point, to
attach a handler and set the desired verbosity.
"""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a simple, readable console logging format for the project.

    Safe to call multiple times: existing handlers on the root logger are
    cleared first, avoiding duplicated log lines - a common issue when a
    notebook cell that calls this is re-run.

    Args:
        level: Root logging level, e.g. ``logging.INFO`` or ``logging.DEBUG``.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(handler)
