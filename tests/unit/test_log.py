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

"""Unit tests of the minimal logging facility (sopa.log.log): the buffered
logger (verbosity, file append, buffer handling on write errors), the
discarding fake logger, and the base class contract.
"""

from pathlib import Path

import pytest

from sopa.log.log import _logger_base, fake_logger, logger


def test_logger_buffers_messages_without_printing(capsys: pytest.CaptureFixture[str]) -> None:
    """Checks that a non-verbose logger buffers messages without console output."""
    lg = logger(verbose=False)
    lg.log_messages("first")
    lg.log_messages("second")
    assert lg.LOG_MESSAGES == ["first", "second"]
    assert capsys.readouterr().out == ""


def test_logger_verbose_prints(capsys: pytest.CaptureFixture[str]) -> None:
    """Checks that a verbose logger echoes each message while buffering it."""
    lg = logger(verbose=True)
    lg.log_messages("hello")
    assert lg.LOG_MESSAGES == ["hello"]
    assert capsys.readouterr().out == "hello\n"


def test_log_write_appends_and_empties_buffer(tmp_path: Path) -> None:
    """Checks that log_write appends the buffered messages to the file and
    empties the buffer, and that a second write appends after the first."""
    lg = logger(verbose=False)
    lg.log_messages("one")
    log_file = tmp_path / "run.log"
    lg.log_write(str(log_file))
    assert lg.LOG_MESSAGES == []
    lg.log_messages("two")
    lg.log_write(str(log_file))
    assert log_file.read_text() == "one\ntwo\n"


def test_log_write_ioerror_keeps_buffer(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Checks that an unwritable destination reports an error and keeps the
    buffer for a later attempt."""
    lg = logger(verbose=False)
    lg.log_messages("kept")
    lg.log_write(str(tmp_path / "missing_dir" / "run.log"))
    assert "ERROR" in capsys.readouterr().out
    assert lg.LOG_MESSAGES == ["kept"]


def test_fake_logger_discards_everything(tmp_path: Path) -> None:
    """Checks that the fake logger accepts messages, never buffers and never
    creates files."""
    lg = fake_logger()
    lg.log_messages("discarded")
    lg.log_write(str(tmp_path / "nothing.log"))
    assert not (tmp_path / "nothing.log").exists()


def test_logger_base_contract() -> None:
    """Checks the base class: construction is not implemented, while the
    message and file methods are no-ops."""
    with pytest.raises(NotImplementedError):
        _logger_base()
    base = object.__new__(_logger_base)
    assert base.log_messages("ignored") is None
    assert base.log_write("nowhere") is None
