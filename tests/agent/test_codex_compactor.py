"""Tests for ``agent.codex_compactor`` and the Codex compaction envelope.

These tests use only ``urllib.request.urlopen`` mocking; no live network
traffic.  The fixture payload mirrors the shape captured against the live
Codex backend: kept user messages followed by one ``compaction`` item.
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest

from agent.codex_compactor import (
    CodexCompactionError,
    CodexNoCompactionError,
    build_compaction_envelope,
    compact_via_codex_responses,
)
from agent.codex_responses_adapter import _chat_messages_to_responses_input


_FIXTURE_OUTPUT = [
    {
        "id": "msg_001",
        "type": "message",
        "status": "completed",
        "content": [{"type": "input_text", "text": "List the files"}],
        "role": "user",
    },
    {
        "id": "msg_002",
        "type": "message",
        "status": "completed",
        "content": [{"type": "input_text", "text": "Now read agent.py"}],
        "role": "user",
    },
    {
        "id": "cmp_001",
        "type": "compaction",
        "encrypted_content": "gAAAAA-encrypted-blob-fixture",
    },
]
_FIXTURE_RESPONSE = {
    "id": "resp_compaction_test",
    "object": "response.compaction",
    "created_at": 1777595244,
    "output": _FIXTURE_OUTPUT,
    "usage": {"input_tokens": 187, "output_tokens": 113, "total_tokens": 300},
}


def _make_urlopen_mock(payload: dict, status: int = 200):
    """Return a patched urlopen factory that yields ``payload`` as JSON."""
    body = json.dumps(payload).encode("utf-8")

    class _FakeResp:
        def __init__(self):
            self.status = status
            self._buf = io.BytesIO(body)

        def read(self):
            return self._buf.read()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    return lambda req, timeout=None: _FakeResp()


class TestCompactViaCodexResponses:
    def test_strips_ids_from_output(self):
        with patch("agent.codex_compactor.urllib.request.urlopen",
                   _make_urlopen_mock(_FIXTURE_RESPONSE)):
            result = compact_via_codex_responses(
                input_items=[
                    {"role": "user", "content": "x"},
                    {"role": "user", "content": "y"},
                ],
                instructions="Be helpful.",
                model="gpt-5.5",
                base_url="https://chatgpt.com/backend-api/codex",
                access_token="fake-jwt",
            )
        for item in result["output"]:
            assert "id" not in item, f"expected id stripped, got {item!r}"

    def test_preserves_compaction(self):
        with patch("agent.codex_compactor.urllib.request.urlopen",
                   _make_urlopen_mock(_FIXTURE_RESPONSE)):
            result = compact_via_codex_responses(
                input_items=[{"role": "user", "content": "x"}],
                instructions="",
                model="gpt-5.5",
                base_url="https://chatgpt.com/backend-api/codex",
                access_token="fake-jwt",
            )
        types = [item.get("type") for item in result["output"]]
        assert "compaction" in types
        compaction = next(i for i in result["output"] if i.get("type") == "compaction")
        assert compaction["encrypted_content"] == "gAAAAA-encrypted-blob-fixture"

    def test_usage_extracted(self):
        with patch("agent.codex_compactor.urllib.request.urlopen",
                   _make_urlopen_mock(_FIXTURE_RESPONSE)):
            result = compact_via_codex_responses(
                input_items=[{"role": "user", "content": "x"}],
                instructions="",
                model="gpt-5.5",
                base_url="https://chatgpt.com/backend-api/codex",
                access_token="fake-jwt",
            )
        assert result["usage"]["total_tokens"] == 300

    def test_empty_input_raises(self):
        with pytest.raises(CodexCompactionError, match="non-empty list"):
            compact_via_codex_responses(
                input_items=[],
                instructions="",
                model="gpt-5.5",
                base_url="https://x",
                access_token="t",
            )

    def test_missing_token_raises(self):
        with pytest.raises(CodexCompactionError, match="access_token"):
            compact_via_codex_responses(
                input_items=[{"role": "user", "content": "x"}],
                instructions="",
                model="gpt-5.5",
                base_url="https://x",
                access_token="",
            )

    def test_missing_compaction_item_raises(self):
        # Server returned only kept items, no compaction.  Caller
        # should treat this as a native no-op instead of falling through to
        # the text-summary path.
        bad_response = {
            "output": [
                {"id": "msg_a", "type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "x"}]}
            ],
            "usage": {"total_tokens": 5},
        }
        with patch("agent.codex_compactor.urllib.request.urlopen",
                   _make_urlopen_mock(bad_response)):
            with pytest.raises(CodexNoCompactionError, match="no compaction"):
                compact_via_codex_responses(
                    input_items=[{"role": "user", "content": "x"}],
                    instructions="",
                    model="gpt-5.5",
                    base_url="https://x",
                    access_token="t",
                )

    def test_http_error_wrapped(self):
        import urllib.error

        def _raise(req, timeout=None):
            raise urllib.error.HTTPError(
                "https://x", 401, "Unauthorized",
                hdrs=None, fp=io.BytesIO(b'{"error":"expired"}'),
            )

        with patch("agent.codex_compactor.urllib.request.urlopen", _raise):
            with pytest.raises(CodexCompactionError) as ei:
                compact_via_codex_responses(
                    input_items=[{"role": "user", "content": "x"}],
                    instructions="",
                    model="gpt-5.5",
                    base_url="https://x",
                    access_token="t",
                )
        assert ei.value.status == 401

    def test_request_target_url(self):
        captured = {}

        def _spy(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["method"] = req.get_method()
            captured["body"] = json.loads(req.data.decode("utf-8"))

            class _Resp:
                status = 200
                def read(self): return json.dumps(_FIXTURE_RESPONSE).encode()
                def __enter__(self): return self
                def __exit__(self, *a): return False

            return _Resp()

        with patch("agent.codex_compactor.urllib.request.urlopen", _spy):
            compact_via_codex_responses(
                input_items=[{"role": "user", "content": "x"}],
                instructions="be helpful",
                model="gpt-5.5",
                base_url="https://chatgpt.com/backend-api/codex",
                access_token="my-token",
            )
        assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses/compact"
        assert captured["method"] == "POST"
        # Authorization header is the bearer token (urllib lowercases header
        # names in header_items()).
        hdrs_lower = {k.lower(): v for k, v in captured["headers"].items()}
        assert hdrs_lower["authorization"] == "Bearer my-token"
        assert captured["body"]["model"] == "gpt-5.5"
        assert captured["body"]["instructions"] == "be helpful"


class TestBuildCompactionEnvelope:
    def test_envelope_contains_side_channel(self):
        envelope = build_compaction_envelope(
            output_items=_FIXTURE_OUTPUT, n_compressed=8
        )
        assert envelope["role"] == "user"
        assert "_codex_responses_items" in envelope
        assert envelope["_codex_responses_items"] is _FIXTURE_OUTPUT
        assert "8" in envelope["content"]


class TestPreflightAcceptsCompaction:
    """Regression: the next /responses call after a compaction must not crash
    on the new compaction item type.  The preflight allowlist used to
    reject any unknown type and would have raised ValueError here."""

    def test_preflight_passes_through_compaction(self):
        from agent.codex_responses_adapter import _preflight_codex_input_items

        items = [
            {"role": "user", "content": "hi"},
            {"type": "compaction", "encrypted_content": "blob123"},
            {"role": "user", "content": "next"},
        ]
        normalized = _preflight_codex_input_items(items)
        types = [n.get("type") for n in normalized]
        assert "compaction" in types
        cs = next(n for n in normalized if n.get("type") == "compaction")
        assert cs["encrypted_content"] == "blob123"
        # IDs must not survive (store=False can't look them up).
        assert "id" not in cs

    def test_preflight_skips_empty_compaction(self):
        from agent.codex_responses_adapter import _preflight_codex_input_items

        items = [
            {"role": "user", "content": "hi"},
            {"type": "compaction", "encrypted_content": ""},
        ]
        normalized = _preflight_codex_input_items(items)
        assert all(n.get("type") != "compaction" for n in normalized)


class TestIterativeCompaction:
    """Regression: iterative compactions must replay prior compaction
    items into the new /compact request, otherwise long-range history is
    silently lost on the second compaction.  codex-rs sends the full state
    (kept items + prior compaction item) back to /compact each time."""

    def test_prior_envelope_expanded_into_next_compact_input(self):
        from agent.codex_compactor import build_compaction_envelope

        prior_envelope = build_compaction_envelope(
            output_items=[
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "old turn"}]},
                {"type": "compaction",
                 "encrypted_content": "blob_v1"},
            ],
            n_compressed=10,
        )
        # New conversation state after first compaction: [system, envelope, fresh user].
        messages = [
            {"role": "system", "content": "be helpful"},
            prior_envelope,
            {"role": "user", "content": "fresh"},
        ]
        items = _chat_messages_to_responses_input(messages)
        # Envelope contributed 2 items + fresh user = 3 (system skipped).
        assert len(items) == 3
        types = [it.get("type") for it in items]
        # The prior compaction must be in the input that the
        # compressor would send to /compact for round 2.  Without this,
        # the server has no knowledge of turns predating the first compaction.
        assert "compaction" in types
        # Sanity: the prior encrypted blob is preserved.
        cs = next(it for it in items if it.get("type") == "compaction")
        assert cs["encrypted_content"] == "blob_v1"


class TestEnvelopeReplayInTransport:
    def test_envelope_emits_items_verbatim(self):
        # The transport must lift the envelope's items into Responses input
        # without going through role-based dispatch.  Items with id should
        # have id stripped (defensive — the compactor strips on insert too).
        envelope = build_compaction_envelope(
            output_items=[
                {"id": "msg_keep_a", "type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "hello"}]},
                {"id": "cmp_X", "type": "compaction",
                 "encrypted_content": "blob"},
            ],
            n_compressed=3,
        )
        # Wrap with a fresh user turn that should follow the envelope.
        messages = [
            {"role": "system", "content": "You are helpful."},
            envelope,
            {"role": "user", "content": "what next?"},
        ]
        items = _chat_messages_to_responses_input(messages)
        # System messages are skipped by the converter; envelope yields its
        # 2 inner items, and the trailing user turn produces 1 item.
        assert len(items) == 3
        assert items[0]["type"] == "message"
        assert items[0].get("role") == "user"
        assert items[1]["type"] == "compaction"
        assert items[1]["encrypted_content"] == "blob"
        # IDs must not have leaked through (store=False can't look them up).
        for item in items[:2]:
            assert "id" not in item
        # Trailing user turn unchanged.
        assert items[2]["role"] == "user"
        assert items[2]["content"] == "what next?"

    def test_empty_envelope_falls_through(self):
        # Defensive: an envelope with empty items list must not skip the
        # message entirely (otherwise a buggy caller would silently lose the
        # display content).  Empty list falls through to normal role dispatch.
        msg = {
            "role": "user",
            "content": "label",
            "_codex_responses_items": [],
        }
        items = _chat_messages_to_responses_input([msg])
        assert items
        assert items[0].get("role") == "user"


class TestContextCompressorCodexPath:
    def test_disabled_by_default(self):
        from agent.context_compressor import ContextCompressor

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="gpt-5.5", quiet_mode=True)
        assert c.codex_native is False

    def test_gate_requires_provider_match(self):
        # codex_native=True but provider!="openai-codex" → gate closed,
        # _compress_via_codex_responses must not be invoked.
        from agent.context_compressor import ContextCompressor

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="gpt-5.5",
                quiet_mode=True,
                codex_native=True,
                provider="openrouter",
            )

        called = {"count": 0}

        def _spy(*a, **kw):
            called["count"] += 1
            return None

        c._compress_via_codex_responses = _spy
        msgs = [{"role": "system", "content": "s"}] + [
            {"role": "user", "content": f"u{i}"} for i in range(20)
        ]
        # Force the text-summary path to also no-op so the test only checks
        # the gate; the LLM call is mocked away.
        with patch.object(c, "_generate_summary", return_value="summary"):
            c.compress(msgs, current_tokens=80000)
        assert called["count"] == 0

    def test_gate_open_for_openai_codex(self):
        from agent.context_compressor import ContextCompressor

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="gpt-5.5",
                quiet_mode=True,
                codex_native=True,
                provider="openai-codex",
            )

        called = {"count": 0}
        sentinel = [{"role": "user", "content": "compacted"}]

        def _spy(messages, current_tokens=None):
            called["count"] += 1
            return sentinel

        c._compress_via_codex_responses = _spy
        msgs = [{"role": "system", "content": "s"}] + [
            {"role": "user", "content": f"u{i}"} for i in range(20)
        ]
        out = c.compress(msgs, current_tokens=80000)
        assert called["count"] == 1
        assert out is sentinel

    def test_native_no_compaction_does_not_fall_back_to_text_summary(self):
        from agent.context_compressor import ContextCompressor

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="gpt-5.5",
                quiet_mode=True,
                codex_native=True,
                provider="openai-codex",
            )

        def _no_compaction(messages, current_tokens=None):
            raise CodexNoCompactionError("no compaction")

        c._compress_via_codex_responses = _no_compaction
        msgs = [{"role": "system", "content": "s"}] + [
            {"role": "user", "content": f"u{i}"} for i in range(20)
        ]

        with patch.object(c, "_generate_summary") as mock_generate:
            out = c.compress(msgs, current_tokens=80000)

        assert out is msgs
        mock_generate.assert_not_called()

    def test_codex_path_falls_back_on_error(self):
        from agent.context_compressor import ContextCompressor

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="gpt-5.5",
                quiet_mode=True,
                codex_native=True,
                provider="openai-codex",
            )

        def _boom(messages, current_tokens=None):
            raise CodexCompactionError("boom")

        c._compress_via_codex_responses = _boom
        msgs = [{"role": "system", "content": "s"}] + [
            {"role": "user", "content": f"u{i}"} for i in range(20)
        ]
        # Text-summary path should run; mock out the LLM call.
        with patch.object(c, "_generate_summary", return_value="text summary"):
            out = c.compress(msgs, current_tokens=80000)
        assert isinstance(out, list)
        # Falls back means the original messages list should be transformed,
        # not returned verbatim.
        assert any("text summary" in str(m.get("content", "")) for m in out)

    def test_native_failure_with_existing_envelope_skips_text_fallback(self):
        """If the history already contains an opaque ``_codex_responses_items``
        envelope (from a prior compaction), a native-compact failure must NOT
        fall through to the text-summary path — that path can only see the
        display label and would irrecoverably collapse the encrypted summary.
        Compression must be skipped instead, returning the messages unchanged.
        """
        from agent.context_compressor import ContextCompressor

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="gpt-5.5",
                quiet_mode=True,
                codex_native=True,
                provider="openai-codex",
            )

        def _boom(messages, current_tokens=None):
            raise CodexCompactionError("token expired")

        c._compress_via_codex_responses = _boom

        envelope = build_compaction_envelope(
            output_items=_FIXTURE_OUTPUT, n_compressed=8
        )
        msgs = [{"role": "system", "content": "s"}, envelope] + [
            {"role": "user", "content": f"u{i}"} for i in range(20)
        ]

        # Spy on the text-summary path: it MUST NOT be called when the
        # history already carries an envelope.
        summary_called = {"count": 0}
        def _spy_summary(*a, **kw):
            summary_called["count"] += 1
            return "text summary"

        with patch.object(c, "_generate_summary", side_effect=_spy_summary):
            out = c.compress(msgs, current_tokens=80000)

        assert summary_called["count"] == 0, (
            "text-summary fallback was invoked despite an existing opaque "
            "envelope — encrypted compaction history would be destroyed"
        )
        # Messages must be returned unchanged so the next compaction attempt
        # can retry against the same payload.
        assert out is msgs or out == msgs


class TestSessionDBEnvelopeRoundtrip:
    """End-to-end DB persistence for the experimental
    ``_codex_responses_items`` envelope (fix/codex-compaction).

    These tests exercise the separate ``codex_compaction_envelopes`` table
    via append_message / replace_messages / get_messages /
    get_messages_as_conversation.  Without this round-trip, a session
    compacted via Codex would be effectively unrecoverable after a
    gateway restart — the marker label survives but the encrypted
    payload needed to replay into the next ``/responses/compact`` call
    would be gone.
    """

    def _make_envelope(self, n_compressed: int = 5):
        return build_compaction_envelope(
            output_items=_FIXTURE_OUTPUT, n_compressed=n_compressed
        )

    def _make_db(self, tmp_path):
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(session_id="s1", source="test")
        return db

    def test_append_message_persists_envelope(self, tmp_path):
        db = self._make_db(tmp_path)
        envelope = self._make_envelope()

        db.append_message(
            session_id="s1",
            role=envelope["role"],
            content=envelope["content"],
            codex_responses_items=envelope["_codex_responses_items"],
        )

        rows = db.get_messages("s1")
        assert len(rows) == 1
        assert rows[0]["role"] == "user"
        assert rows[0]["_codex_responses_items"] == envelope["_codex_responses_items"]

    def test_get_messages_as_conversation_restores_envelope(self, tmp_path):
        db = self._make_db(tmp_path)
        envelope = self._make_envelope()

        db.append_message(
            session_id="s1",
            role=envelope["role"],
            content=envelope["content"],
            codex_responses_items=envelope["_codex_responses_items"],
        )
        db.append_message(
            session_id="s1", role="user", content="follow up?",
        )

        convo = db.get_messages_as_conversation("s1")
        assert len(convo) == 2
        assert convo[0]["_codex_responses_items"] == envelope["_codex_responses_items"]
        # Trailing turn is a plain user message — no envelope key.
        assert "_codex_responses_items" not in convo[1]

    def test_replace_messages_roundtrips_envelope(self, tmp_path):
        db = self._make_db(tmp_path)
        envelope = self._make_envelope()

        # Seed with non-envelope messages first.
        db.append_message(session_id="s1", role="user", content="old1")
        db.append_message(session_id="s1", role="assistant", content="old2")

        # Replace with [system, envelope, follow-up].
        new_messages = [
            {"role": "system", "content": "S"},
            envelope,
            {"role": "user", "content": "next"},
        ]
        db.replace_messages("s1", new_messages)

        rows = db.get_messages("s1")
        assert len(rows) == 3
        assert rows[1]["_codex_responses_items"] == envelope["_codex_responses_items"]

    def test_replace_messages_cascades_envelope_cleanup(self, tmp_path):
        """``replace_messages`` deletes all messages for the session before
        re-inserting; ON DELETE CASCADE on ``codex_compaction_envelopes``
        must remove the orphan envelope rows so they don't pile up."""
        db = self._make_db(tmp_path)
        envelope = self._make_envelope()

        # Seed with an envelope-bearing message.
        db.append_message(
            session_id="s1",
            role=envelope["role"],
            content=envelope["content"],
            codex_responses_items=envelope["_codex_responses_items"],
        )

        # Now replace with a plain message list — old envelope must vanish.
        db.replace_messages("s1", [{"role": "user", "content": "fresh"}])

        rows = db.get_messages("s1")
        assert len(rows) == 1
        assert "_codex_responses_items" not in rows[0] or not rows[0].get(
            "_codex_responses_items"
        )

        # Verify the envelope table itself is empty for this session's row.
        with db._lock:
            cur = db._conn.execute(
                "SELECT COUNT(*) FROM codex_compaction_envelopes"
            )
            assert cur.fetchone()[0] == 0

    def test_iterative_compaction_replays_through_db(self, tmp_path):
        """The point of native compaction is that a *second* compaction can
        re-feed the prior ``compaction`` back into ``/responses/compact``.
        After a DB round-trip, the envelope must still expand into the
        correct Responses items via ``_chat_messages_to_responses_input``.
        """
        db = self._make_db(tmp_path)
        envelope = self._make_envelope()

        db.append_message(
            session_id="s1",
            role=envelope["role"],
            content=envelope["content"],
            codex_responses_items=envelope["_codex_responses_items"],
        )
        db.append_message(session_id="s1", role="user", content="next turn")

        convo = db.get_messages_as_conversation("s1")
        items = _chat_messages_to_responses_input(convo)

        # Envelope expanded inline: the compaction item must be in
        # the Responses input for the next /compact call.
        assert any(it.get("type") == "compaction" for it in items)
        # The trailing user turn is also there.
        assert items[-1].get("role") == "user"
