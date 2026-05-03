"""Offline validation tests for request bounds and admin fixture names."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    MAX_MESSAGE_CHARS,
    MAX_QUERY_CHARS,
    RecallRequest,
    RunFixtureRequest,
    TurnMessage,
)


def test_recall_query_has_local_length_bound():
    with pytest.raises(ValidationError):
        RecallRequest(
            query="x" * (MAX_QUERY_CHARS + 1),
            session_id="session-1",
            user_id="user-1",
        )


def test_turn_message_content_has_local_length_bound():
    with pytest.raises(ValidationError):
        TurnMessage(role="user", content="x" * (MAX_MESSAGE_CHARS + 1))


def test_fixture_name_rejects_path_traversal():
    with pytest.raises(ValidationError):
        RunFixtureRequest(name="../conversations")


def test_fixture_name_allows_known_fixture_slug():
    assert RunFixtureRequest(name="v2").name == "v2"
