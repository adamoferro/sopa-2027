#!/usr/bin/python3

#   Copyright 2016-2019 Adamo Ferro
#
#   This file is part of SOPA.
#
#   SOPA is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   SOPA is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with SOPA. If not, see <http://www.gnu.org/licenses/>.
#
#   The use of SOPA or part of it for the creation of any sub-product
#   (e.g., scientific papers, posters, images, other softwares)
#   must be acknowledged.

"""Minimal logging: loggers accumulate messages in a list, optionally echo
them to the console, and append them to a text file on request.
"""

class _logger_base():
    """Interface of the loggers used by the dataset readers and the processing classes."""

    def __init__(self, verbose: bool = False) -> None:
        raise NotImplementedError

    def log_messages(self, new_message: str) -> None:
        """Handles a new message (no-op in the base class)."""
        pass

    def log_write(self, filename: str) -> None:
        """Writes the buffered messages to a file (no-op in the base class)."""
        pass


class logger(_logger_base):
    """Collects log messages in memory, printing each of them when verbose, and appends the buffer to a text file with `log_write`."""

    def __init__(self, verbose: bool = False) -> None:
        self.LOG_MESSAGES = list()
        self.verbose = verbose

    def log_messages(self, new_message: str) -> None:
        """Appends the message to the in-memory buffer and prints it when verbose."""
        self.LOG_MESSAGES.append(new_message)
        if self.verbose:
            print(new_message)

    def log_write(self, filename: str) -> None:
        """Appends the buffered messages to the given text file and empties the buffer."""
        try:
            with open(filename, 'a') as fp:
                for lm in self.LOG_MESSAGES:
                    fp.write(f'{lm}\n')
                self.LOG_MESSAGES = list()           # empty for next write
        except IOError:
            print("ERROR: cannot write log file or append new messages to existing log file.")


class fake_logger(_logger_base):
    """Discards every message; used as the default logger when none is supplied."""

    def __init__(self, verbose: bool = False) -> None:
        pass
