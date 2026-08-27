import logging
from logging import Logger
from typing import Self

import colorlog


class LoggerSingleton:
    """Lazily creates, and then always returns, the single 'artc' logger."""

    _instance: Self | None = None
    # Always set by '_setup_logger', called from '__new__' right after the only
    # instance is created. basedpyright only recognizes assignments made in the
    # class body or in '__init__' as initialization, neither of which applies
    # to this '__new__' based singleton.
    logger: Logger  # pyright: ignore[reportUninitializedInstanceVariable]

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup_logger()
        return cls._instance

    def _setup_logger(self) -> None:
        """Configures the 'artc' logger with a colored console handler."""
        self.logger = logging.getLogger("artc")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)

            formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(levelname)s - %(message)s",
                log_colors={
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red",
                },
            )

            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def get_logger(self) -> Logger:
        """Returns the shared 'artc' logger."""
        return self.logger
