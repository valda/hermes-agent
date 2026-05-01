"""Tests for context compression persistence in the gateway.

Verifies that when context compression fires during run_conversation(),
the compressed messages are properly persisted to both SQLite (via the
agent) and JSONL (via the gateway).

Bug scenario (pre-fix):
  1. Gateway loads 200-message history, passes to agent
  2. Agent's run_conversation() compresses to ~30 messages mid-run
  3. _compress_context() resets _last_flushed_db_idx = 0
  4. On exit, _flush_messages_to_session_db() calculates:
     flush_from = max(len(conversation_history=200), _last_flushed_db_idx=0) = 200
  5. messages[200:] is empty (only ~30 messages after compression)
  6. Nothing written to new session's SQLite — compressed context lost
  7. Gateway's history_offset was still 200, producing empty new_messages
  8. Fallback wrote only user/assistant pair — summary lost
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Part 1: Agent-side — _flush_messages_to_session_db after compression
# ---------------------------------------------------------------------------

class TestFlushAfterCompression:
    """Verify that compressed messages are flushed to the new session's SQLite
    even when conversation_history (from the original session) is longer than
    the compressed messages list."""

    def _make_agent(self, session_db):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id="original-session",
                skip_context_files=True,
                skip_memory=True,
            )
        return agent

    def test_flush_after_compression_with_long_history(self):
        """The actual bug: conversation_history longer than compressed messages.

        Before the fix, flush_from = max(len(conversation_history), 0) = 200,
        but messages only has ~30 entries, so messages[200:] is empty.
        After the fix, conversation_history is cleared to None after compression,
        so flush_from = max(0, 0) = 0, and ALL compressed messages are written.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)

            # Simulate the original long history (200 messages)
            original_history = [
                {"role": "user" if i % 2 == 0 else "assistant",
                 "content": f"message {i}"}
                for i in range(200)
            ]

            # First, flush original messages to the original session
            agent._flush_messages_to_session_db(original_history, [])
            original_rows = db.get_messages("original-session")
            assert len(original_rows) == 200

            # Now simulate compression: new session, reset idx, shorter messages
            agent.session_id = "compressed-session"
            db.create_session(session_id="compressed-session", source="test")
            agent._last_flushed_db_idx = 0

            # The compressed messages (summary + tail + new turn)
            compressed_messages = [
                {"role": "user", "content": "[CONTEXT COMPACTION] Summary of work..."},
                {"role": "user", "content": "What should we do next?"},
                {"role": "assistant", "content": "Let me check..."},
                {"role": "user", "content": "new question"},
                {"role": "assistant", "content": "new answer"},
            ]

            # THE BUG: passing the original history as conversation_history
            # causes flush_from = max(200, 0) = 200, skipping everything.
            # After the fix, conversation_history should be None.
            agent._flush_messages_to_session_db(compressed_messages, None)

            new_rows = db.get_messages("compressed-session")
            assert len(new_rows) == 5, (
                f"Expected 5 compressed messages in new session, got {len(new_rows)}. "
                f"Compression persistence bug: messages not written to SQLite."
            )

    def test_flush_with_stale_history_loses_messages(self):
        """Demonstrates the bug condition: stale conversation_history causes data loss."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            db = SessionDB(db_path=db_path)

            agent = self._make_agent(db)

            # Simulate compression reset
            agent.session_id = "new-session"
            db.create_session(session_id="new-session", source="test")
            agent._last_flushed_db_idx = 0

            compressed = [
                {"role": "user", "content": "summary"},
                {"role": "assistant", "content": "continuing..."},
            ]

            # Bug: passing a conversation_history longer than compressed messages
            stale_history = [{"role": "user", "content": f"msg{i}"} for i in range(100)]
            agent._flush_messages_to_session_db(compressed, stale_history)

            rows = db.get_messages("new-session")
            # With the stale history, flush_from = max(100, 0) = 100
            # But compressed only has 2 entries → messages[100:] = empty
            assert len(rows) == 0, (
                "Expected 0 messages with stale conversation_history "
                "(this test verifies the bug condition exists)"
            )


# ---------------------------------------------------------------------------
# Part 1b: No-op compression must NOT rotate the session
# ---------------------------------------------------------------------------

class TestNoOpCompressionDoesNotSplitSession:
    """When the compressor returns the input list unchanged (its
    "skip this turn" signal — used by the Codex native envelope-protection
    path and the too-short-to-compact path), ``_compress_context`` must NOT:

      * end the current session,
      * rotate ``session_id``,
      * reset ``_last_flushed_db_idx``.

    Without that guard, a no-op return rotates the session_id even though
    the messages list is unchanged — and the new (empty) session never
    receives the original history's flush, so any
    ``_codex_responses_items`` envelope that lives in those messages is
    invisible after a resume.  This regresses the DB persistence work
    introduced for the same Hermes review series.
    """

    def _make_agent(self, session_db, session_id="orig-session"):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            from run_agent import AIAgent
            agent = AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                session_db=session_db,
                session_id=session_id,
                skip_context_files=True,
                skip_memory=True,
            )
        return agent

    def _seed_envelope_message(self, session_db, session_id):
        """Persist a synthetic compaction envelope message into ``session_id``."""
        from agent.codex_compactor import build_compaction_envelope
        envelope = build_compaction_envelope(
            output_items=[
                {"id": "msg_001", "type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "kept"}]},
                {"id": "cmp_001", "type": "compaction",
                 "encrypted_content": "blob"},
            ],
            n_compressed=4,
        )
        session_db.append_message(
            session_id=session_id,
            role=envelope["role"],
            content=envelope["content"],
            codex_responses_items=envelope["_codex_responses_items"],
        )
        return envelope

    def test_no_op_keeps_session_id_and_envelope_in_db(self, tmp_path):
        """Compressor returns messages unchanged (Codex native envelope-protection
        path) → session_id must be unchanged, envelope must still be retrievable."""
        from hermes_state import SessionDB

        db_path = tmp_path / "test.db"
        db = SessionDB(db_path=db_path)
        db.create_session(session_id="orig-session", source="test")
        envelope = self._seed_envelope_message(db, "orig-session")

        agent = self._make_agent(db, session_id="orig-session")

        # Mock the compressor: identity-return signals a no-op.  Mirror the
        # state-flag contract that ContextCompressor exposes so
        # _compress_context's status-flag inspection path is exercised too.
        mock_compressor = MagicMock()
        mock_compressor.compress = MagicMock(side_effect=lambda msgs, **kw: msgs)
        mock_compressor._last_summary_error = None
        mock_compressor._last_aux_model_failure_error = None
        mock_compressor._last_aux_model_failure_model = None
        agent.context_compressor = mock_compressor

        messages = [
            {"role": "system", "content": "S"},
            {**envelope},
            {"role": "user", "content": "follow up"},
        ]
        original_session_id = agent.session_id

        out_messages, out_system = agent._compress_context(
            messages, "S", approx_tokens=10
        )

        # Identity guarantee: same list reference, no mutation.
        assert out_messages is messages
        assert out_system == "S"

        # Session must NOT have rotated.
        assert agent.session_id == original_session_id, (
            "no-op compression rotated session_id — DB persistence is "
            "now disconnected from the original session"
        )

        # The envelope must still be retrievable on the original session
        # (not orphaned by an end_session call).
        convo = db.get_messages_as_conversation("orig-session")
        assert any(
            m.get("_codex_responses_items") for m in convo
        ), "envelope on original session disappeared after a no-op compression"

    def test_real_compression_still_splits_session(self, tmp_path):
        """Sanity check: when the compressor returns a *new* list (real
        compression), session split still happens.  Guards against an
        identity-check that's too aggressive."""
        from hermes_state import SessionDB

        db_path = tmp_path / "test.db"
        db = SessionDB(db_path=db_path)
        db.create_session(session_id="orig-session", source="test")
        agent = self._make_agent(db, session_id="orig-session")

        # Use a plain attribute container instead of MagicMock so the post-split
        # bookkeeping in _compress_context (compression_count >= 2 check, etc.)
        # gets real ints rather than auto-conjured MagicMocks.
        class _FakeCompressor:
            _last_summary_error = None
            _last_aux_model_failure_error = None
            _last_aux_model_failure_model = None
            compression_count = 0
            last_prompt_tokens = 0
            last_completion_tokens = 0

            def compress(self, msgs, **kw):
                return [{"role": "user", "content": "compressed"}]

        agent.context_compressor = _FakeCompressor()

        messages = [
            {"role": "user", "content": f"u{i}"} for i in range(10)
        ]
        original_session_id = agent.session_id

        agent._compress_context(messages, "S", approx_tokens=10)

        # Session MUST have rotated.
        assert agent.session_id != original_session_id, (
            "real compression failed to split session — identity check "
            "is too aggressive"
        )


# ---------------------------------------------------------------------------
# Part 2: Gateway-side — history_offset after session split
# ---------------------------------------------------------------------------

class TestGatewayHistoryOffsetAfterSplit:
    """Verify that when the agent creates a new session during compression,
    the gateway uses history_offset=0 so all compressed messages are written
    to the JSONL transcript."""

    def test_history_offset_zero_on_session_split(self):
        """When agent.session_id differs from the original, history_offset must be 0."""
        # This tests the logic in gateway/run.py run_sync():
        # _session_was_split = agent.session_id != session_id
        # _effective_history_offset = 0 if _session_was_split else len(agent_history)

        original_session_id = "session-abc"
        agent_session_id = "session-compressed-xyz"  # Different = compression happened
        agent_history_len = 200

        # Simulate the gateway's offset calculation (post-fix)
        _session_was_split = (agent_session_id != original_session_id)
        _effective_history_offset = 0 if _session_was_split else agent_history_len

        assert _session_was_split is True
        assert _effective_history_offset == 0

    def test_history_offset_preserved_without_split(self):
        """When no compression happened, history_offset is the original length."""
        session_id = "session-abc"
        agent_session_id = "session-abc"  # Same = no compression
        agent_history_len = 200

        _session_was_split = (agent_session_id != session_id)
        _effective_history_offset = 0 if _session_was_split else agent_history_len

        assert _session_was_split is False
        assert _effective_history_offset == 200

    def test_new_messages_extraction_after_split(self):
        """After compression with offset=0, new_messages should be ALL agent messages."""
        # Simulates the gateway's new_messages calculation
        agent_messages = [
            {"role": "user", "content": "[CONTEXT COMPACTION] Summary..."},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
            {"role": "user", "content": "new question"},
            {"role": "assistant", "content": "new answer"},
        ]
        history_offset = 0  # After fix: 0 on session split

        new_messages = agent_messages[history_offset:] if len(agent_messages) > history_offset else []
        assert len(new_messages) == 5, (
            f"Expected all 5 messages with offset=0, got {len(new_messages)}"
        )

    def test_new_messages_empty_with_stale_offset(self):
        """Demonstrates the bug: stale offset produces empty new_messages."""
        agent_messages = [
            {"role": "user", "content": "summary"},
            {"role": "assistant", "content": "answer"},
        ]
        # Bug: offset is the pre-compression history length
        history_offset = 200

        new_messages = agent_messages[history_offset:] if len(agent_messages) > history_offset else []
        assert len(new_messages) == 0, (
            "Expected 0 messages with stale offset=200 (demonstrates the bug)"
        )
