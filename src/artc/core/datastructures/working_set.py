import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from audioread.exceptions import DecodeError

from .. import errors

logger = errors.logger_config.LoggerSingleton().get_logger()


@dataclass
class AudioFile:
    path: Path
    name: str
    audio_signal_unloaded: Callable[[], np.ndarray]
    sample_rate: int

    @property
    def audio_signal_loaded(self) -> np.ndarray:
        return self.audio_signal_unloaded()

    def check_audio(self, configuration_path: Path) -> bool:
        verifications = [
            # (Check function,
            # {Function parameters},
            # Error message)
            # check_audio_format already re-checks corruption internally, so
            # there is no separate check_audio_corruption entry here. Adding
            # one would just read the file a second time for the same result.
            (
                errors.check_audio_format,
                {
                    "path": self.path,
                    "name": self.name,
                    "configuration_path": configuration_path,
                },
                f"Invalid file format for '{self.name}'",
            ),
            (
                errors.check_path_accessible,
                {"path": self.path},
                f"Path '{self.path}' does not exist or is not accessible",
            ),
            (
                errors.check_path_accessible,
                {"path": configuration_path.parent},
                f"Path '{configuration_path.parent}' does not exist or is not accessible",
            ),
        ]

        # all() short-circuits on the first failing check: each item runs
        # check_function, logs error_message and contributes False if it fails,
        # or True if it passes.
        no_error = all(
            logger.error(error_message) if not check_function(**kwargs) else True
            for check_function, kwargs, error_message in verifications
        )

        return no_error


class WorkingSet:
    name: str

    def __init__(
        self,
        name: str,
        /,
        *,
        test_mode: bool = False,
        data_set: dict | None = None,
    ):
        self.name = name

        if not test_mode:
            self.working_set = {"individual_files": []}
        else:
            self.working_set = data_set if data_set is not None else {"individual_files": []}

    def __getitem__(self, item: str | tuple[str, str], /) -> AudioFile | None:
        if isinstance(item, tuple):
            name, group = item
        else:
            name, group = item, "individual_files"

        if group not in self.working_set:
            logger.error(
                f"No group with name '{group}' was found in working set '{self.name}'"
            )
            return None

        for file in self.working_set[group]:
            if file.name == name:
                return file
        logger.error(
            f"No file with name '{name}' was found in group '{group}' in working set "
            f"'{self.name}'"
        )
        return None

    def __contains__(self, item: str | tuple[str, str], /) -> bool:
        """Check whether a file exists in the working set, supporting the `in` operator

        Supports two calling forms via the `in` operator:
            - `"file.mp3" in working_set` checks the default group
              ("individual_files")
            - `("file.mp3", "favorites") in working_set` checks a specific
              group, passed alongside the file name as a tuple

        Args:
            item:
                Either a file name (str), checked against the default group,
                or a (name, group) tuple, checked against the given group

        Returns:
            True if a file with the given name exists in the given group,
            False otherwise (including when the group itself does not exist)
        """
        if isinstance(item, tuple):
            name, group = item
        else:
            name, group = item, "individual_files"

        return group in self.working_set and name in [
            audio.name for audio in self.working_set[group]
        ]

    def add_file(
        self,
        *,
        path: Path,
        name: str,
        configuration_path: Path,
        group: str = "individual_files",
    ) -> bool:
        if group == "":
            logger.error("Can not add groups with empty names")
            return False

        if not errors.validate_path(path=path, name=name):
            logger.error(f"Path '{path / name}' does not exist or is not accessible")
            return False

        try:
            # DecodeError: audioread (librosa's fallback backend) failed to decode
            # the file (e.g. corrupted content, unsupported codec).
            # OSError: raised directly by some audioread backends (ffdec, macca)
            # for a missing/unreadable file, bypassing DecodeError entirely.
            # EOFError: raised by audioread for a zero byte or truncated file.
            audio_signal, sample_rate = librosa.load(path / name)
        except (DecodeError, OSError, EOFError) as e:
            logger.error(f"Could not load audio file '{name}': {e}")
            return False

        audio = AudioFile(
            path=path,
            name=name,
            audio_signal_unloaded=lambda: audio_signal,
            sample_rate=int(sample_rate),
        )

        if not audio.check_audio(configuration_path):
            logger.error(
                f"Could not add file '{name}' in group '{group}' in working set "
                f"'{self.name}'"
            )
            return False

        if group in self.working_set:
            self.working_set[group].append(audio)
        else:
            self.working_set[group] = [audio]
        return True

    def remove_file(self, *, name: str, group: str = "individual_files") -> bool:
        if group not in self.working_set or not any(
            audio.name == name for audio in self.working_set[group]
        ):
            logger.error(
                f"Could not delete file. "
                f"No file with name '{name}' was found in key '{group}' in working set "
                f"'{self.name}'"
            )
            return False
        else:
            self.working_set[group] = [
                audio for audio in self.working_set[group] if audio.name != name
            ]
            return True

    def add_directory(
        self, *, path: Path, configuration_path: Path, group: str = "individual_files"
    ) -> bool:
        any_files_added = False

        directory_verifications = [
            (
                errors.check_path_accessible,
                (path,),
                f"Path '{path}' does not exist or is not accessible",
            ),
            (
                errors.check_path_accessible,
                (configuration_path.parent,),
                f"Path '{configuration_path.parent}' does not exist or is not accessible",
            ),
            (
                lambda check_group: group != "",
                (group,),
                "Can not add groups with empty names",
            ),
        ]

        no_error = all(
            logger.error(error_message) if not check_function(*args) else True
            for check_function, args, error_message in directory_verifications
        )

        if not no_error or path.as_posix() == ".":
            return False

        for file_name in sorted(os.listdir(path), key=str.lower):
            if os.path.isfile(os.path.join(path, file_name)) and self.add_file(
                path=path,
                name=file_name,
                configuration_path=configuration_path,
                group=group,
            ):
                any_files_added = True

        return any_files_added
