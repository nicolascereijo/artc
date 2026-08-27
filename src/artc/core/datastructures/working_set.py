import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt
from audioread.exceptions import DecodeError

from ..errors import (
    check_audio_format,
    check_path_accessible,
    logger_config,
    validate_path,
)

logger = logger_config.LoggerSingleton().get_logger()


@dataclass
class AudioFile:
    """Holds one audio file's identity, location and decoded signal.

    'audio_signal_unloaded' defers decoding the file until
    'audio_signal_loaded' is first accessed, since a 'WorkingSet' can hold many
    files and not every caller needs the decoded signal.

    Attributes:
        path: Directory containing the audio file.
        name: File name, including its extension.
        audio_signal_unloaded: Zero argument callable returning the decoded
            audio signal.
        sample_rate: Sampling rate, in Hz, of the decoded signal.
    """

    path: Path
    name: str
    audio_signal_unloaded: Callable[[], npt.NDArray[np.float32]]
    sample_rate: int

    @property
    def audio_signal_loaded(self) -> npt.NDArray[np.float32]:
        """Decodes, or returns the already decoded, audio signal."""
        return self.audio_signal_unloaded()

    def check_audio(self, configuration_path: Path) -> bool:
        """Runs every audio validation check and logs each failure.

        Args:
            configuration_path: Path to the TOML configuration file used to
                validate the audio format against.

        Returns:
            'True' if every check passes, 'False' if any check fails.
        """
        # 'check_audio_format' already checks corruption again internally, so
        # there is no separate 'check_audio_corruption' entry here. Adding one
        # would just read the file a second time for the same result.
        verifications: list[tuple[Callable[[], bool], str]] = [
            (
                lambda: check_audio_format(
                    path=self.path,
                    name=self.name,
                    configuration_path=configuration_path,
                ),
                f"Invalid file format for '{self.name}'",
            ),
            (
                lambda: check_path_accessible(path=self.path),
                f"Path '{self.path}' does not exist or is not accessible",
            ),
            (
                lambda: check_path_accessible(
                    path=configuration_path.parent
                ),
                f"Path '{configuration_path.parent}' does not exist or " +
                "is not accessible",
            ),
        ]

        # 'all' short circuits on the first failing check, each entry runs
        # its check and, if it fails, logs its message. A failing entry
        # yields a falsy value, a passing one yields 'True'.
        no_error = all(
            logger.error(message) if not check() else True
            for check, message in verifications
        )

        return no_error


class WorkingSet:
    """A named collection of audio files, grouped by an arbitrary key.

    Files are grouped under string keys, defaulting to 'individual_files' when
    no group is given. This lets callers keep unrelated collections of audio
    inside the same 'WorkingSet', for example one group per labeled category
    being compared.

    Attributes:
        name: Name identifying this working set.
        working_set: Maps each group name to the list of 'AudioFile'
            objects it holds.
    """

    name: str
    working_set: dict[str, list[AudioFile]]

    def __init__(
        self,
        name: str,
        /,
        *,
        test_mode: bool = False,
        data_set: dict[str, list[AudioFile]] | None = None,
    ):
        """Creates an empty working set, or wraps an existing one for tests.

        Args:
            name: Name identifying this working set.
            test_mode: If 'True', 'data_set' is used as the initial working set
                instead of an empty one, letting tests set up a specific state
                directly.
            data_set: Initial working set, only used when 'test_mode' is
                'True'. Defaults to an empty working set.
        """
        self.name = name

        if not test_mode:
            self.working_set = {"individual_files": []}
        else:
            self.working_set = (
                data_set if data_set is not None else {"individual_files": []}
            )

    def __getitem__(self, item: str | tuple[str, str], /) -> AudioFile | None:
        """Looks up a file by name, optionally within a specific group.

        Args:
            item: Either a file name, looked up in the default group, or a
                (name, group) tuple, looked up in the given group.

        Returns:
            The matching 'AudioFile', or 'None' if the group does not exist or
            no file with that name is in it.
        """
        if isinstance(item, tuple):
            name, group = item
        else:
            name, group = item, "individual_files"

        if group not in self.working_set:
            logger.error(
                f"No group with name '{group}' was found in working set " +
                f"'{self.name}'"
            )
            return None

        for file in self.working_set[group]:
            if file.name == name:
                return file
        logger.error(
            f"No file with name '{name}' was found in group '{group}' in " +
            f"working set '{self.name}'"
        )
        return None

    def __contains__(self, item: str | tuple[str, str], /) -> bool:
        """Checks whether a file exists in the working set, supporting the 'in'
        operator.

        Supports two calling forms via the 'in' operator. '"file.mp3" in
        working_set' checks the default group ('individual_files'), and
        '("file.mp3", "favorites") in working_set' checks a specific group,
        passed alongside the file name as a tuple.

        Args:
            item: Either a file name, checked against the default group, or a
                (name, group) tuple, checked against the given group.

        Returns:
            'True' if a file with the given name exists in the given group,
            'False' otherwise, including when the group itself does not exist.
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
        """Loads, validates and adds one audio file to the working set.

        Args:
            path: Directory containing the audio file.
            name: File name, including its extension.
            configuration_path: Path to the TOML configuration file used to
                validate the audio format against.
            group: Group to add the file to, created if it does not already
                exist.

        Returns:
            'True' if the file was loaded, validated and added, 'False'
            otherwise, with the reason logged.
        """
        if group == "":
            logger.error("Can not add groups with empty names")
            return False

        if not validate_path(path=path, name=name):
            logger.error(
                f"Path '{path / name}' does not exist or is not accessible"
            )
            return False

        try:
            # DecodeError: audioread (librosa's fallback backend) failed to
            # decode the file, for example corrupted content or an unsupported
            # codec.
            # OSError: raised directly by some audioread backends (ffdec,
            # macca) for a missing or unreadable file, bypassing DecodeError
            # entirely.
            # EOFError: raised by audioread for a zero byte or truncated file.
            audio_signal, sample_rate = librosa.load(path / name)
        except (DecodeError, OSError, EOFError) as e:
            logger.error(f"Could not load audio file '{name}': {e}")
            return False

        def audio_signal_unloaded() -> npt.NDArray[np.float32]:
            return audio_signal

        audio = AudioFile(
            path=path,
            name=name,
            audio_signal_unloaded=audio_signal_unloaded,
            sample_rate=int(sample_rate),
        )

        if not audio.check_audio(configuration_path):
            logger.error(
                f"Could not add file '{name}' in group '{group}' in " +
                f"working set '{self.name}'"
            )
            return False

        if group in self.working_set:
            self.working_set[group].append(audio)
        else:
            self.working_set[group] = [audio]
        return True

    def remove_file(
        self, *, name: str, group: str = "individual_files",
    ) -> bool:
        """Removes one file from a group by name.

        Args:
            name: File name of the file to remove.
            group: Group to remove the file from.

        Returns:
            'True' if a matching file was found and removed, 'False' otherwise,
            with the reason logged.
        """
        if group not in self.working_set or not any(
            audio.name == name for audio in self.working_set[group]
        ):
            logger.error(
                f"Could not delete file, no file with name '{name}' was " +
                f"found in group '{group}' in working set '{self.name}'"
            )
            return False
        else:
            self.working_set[group] = [
                audio
                for audio in self.working_set[group]
                if audio.name != name
            ]
            return True

    def add_directory(
        self,
        *,
        path: Path,
        configuration_path: Path,
        group: str = "individual_files",
    ) -> bool:
        """Adds every audio file directly inside a directory to a group.

        Files are added in alphabetical order, sorted case insensitively.
        Subdirectories are not recursed into.

        Args:
            path: Directory to add every audio file from.
            configuration_path: Path to the TOML configuration file used to
                validate each file's format against.
            group: Group to add the files to, created if it does not already
                exist.

        Returns:
            'True' if at least one file was added, 'False' otherwise.
        """
        any_files_added = False

        directory_verifications: list[tuple[Callable[[], bool], str]] = [
            (
                lambda: check_path_accessible(path=path),
                f"Path '{path}' does not exist or is not accessible",
            ),
            (
                lambda: check_path_accessible(
                    path=configuration_path.parent
                ),
                f"Path '{configuration_path.parent}' does not exist or " +
                "is not accessible",
            ),
            (
                lambda: group != "",
                "Can not add groups with empty names",
            ),
        ]

        no_error = all(
            logger.error(message) if not check() else True
            for check, message in directory_verifications
        )

        if not no_error or path.as_posix() == ".":
            return False

        for file_name in sorted(os.listdir(path), key=str.lower):
            if os.path.isfile(
                os.path.join(path, file_name)
            ) and self.add_file(
                path=path,
                name=file_name,
                configuration_path=configuration_path,
                group=group,
            ):
                any_files_added = True

        return any_files_added
