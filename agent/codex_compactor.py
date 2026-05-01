"""Native Codex Responses API compaction.

The OpenAI Codex backend (``chatgpt.com/backend-api/codex``) exposes a unary
``POST /responses/compact`` endpoint that returns an opaque ``compaction``
item plus a small set of verbatim items the server chose to keep (typically the
user-side messages).  This module is a thin Python port of codex-rs's
``compact_conversation_history`` so Hermes can use the same path instead of (or
in addition to) the cross-provider text-summary compactor in
``agent.context_compressor``.

The compaction output is opaque — only the same Codex backend can decrypt
``encrypted_content`` on a follow-up ``/responses`` call.  Callers must therefore
ensure the *next* inference also runs through the Codex transport on the same
account; switching providers mid-session loses the compacted history.

Contract — head/tail protection is delegated to the server
----------------------------------------------------------
This path is fundamentally different from the text-summary compactor in
``agent.context_compressor``.  In codex-rs the remote-compact path
(``run_remote_compact_task``) sends the **full** conversation history to
``/responses/compact`` and lets the **server** decide which user turns to
keep verbatim and where to insert the opaque ``compaction`` item.
The text-summary path's client-side **truncation / boundary** invariants —

  * ``protect_first_n`` (head messages excluded from summarisation),
  * ``protect_last_n`` / ``tail_token_budget`` (tail messages excluded),
  * "the latest user message must remain in the tail",
  * orphaned tool-call / tool-result pair *cleanup that arises from those cuts*

**do not apply** here.  We do not pre-trim or rewrite the transcript to
enforce head/tail preservation; the full Responses-shaped conversation is
sent to the server and the Codex backend owns the keep policy.

(This does **not** mean tool-call/tool-result well-formedness can be
ignored: the request must still be a valid Responses input, and
``_chat_messages_to_responses_input`` is responsible for shaping it.
What goes away is the *cut-induced* orphaning that the text-summary
path's ``_align_boundary_forward`` / pair-sanitisation logic exists to
repair.)

Second-guessing keep policy client-side would diverge from upstream's
contract and break iterative compaction (where a prior
``compaction`` is replayed back into the next ``/compact`` call).

If a future change wants to constrain server keep policy, the right place
is a server-side feature request, not a client-side pre-trim.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


CODEX_COMPACT_PATH = "/responses/compact"
DEFAULT_TIMEOUT = 60.0


class CodexCompactionError(RuntimeError):
    """Raised when ``/responses/compact`` returns an error or unexpected payload."""

    def __init__(self, message: str, *, status: Optional[int] = None, body: Optional[str] = None):
        super().__init__(message)
        self.status = status
        self.body = body


class CodexNoCompactionError(CodexCompactionError):
    """Raised when ``/responses/compact`` succeeds but returns no compaction."""


def _strip_ids(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove ``id`` from each item.

    The Responses API with ``store=False`` cannot look up items by id and
    returns 404 if a replayed item carries one.  The compaction output we
    receive includes ``msg_*`` and ``cmp_*`` ids that are only meaningful in
    the server-side store; strip them before persisting for replay.
    """
    cleaned: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cleaned.append({k: v for k, v in item.items() if k != "id"})
    return cleaned


def compact_via_codex_responses(
    *,
    input_items: List[Dict[str, Any]],
    instructions: str,
    model: str,
    base_url: str,
    access_token: str,
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Call ``POST {base_url}/responses/compact`` and return the parsed body.

    Returns ``{"output": [...id-stripped items...], "usage": {...}, "raw": {...}}``.

    ``input_items`` must already be in Responses API shape (the same shape the
    Codex transport sends to ``/responses``).  Use
    ``agent.codex_responses_adapter._chat_messages_to_responses_input`` to
    convert chat-style messages first.
    """
    if not isinstance(input_items, list) or not input_items:
        raise CodexCompactionError("input_items must be a non-empty list")
    if not isinstance(access_token, str) or not access_token.strip():
        raise CodexCompactionError("access_token is required")

    url = base_url.rstrip("/") + CODEX_COMPACT_PATH
    payload: Dict[str, Any] = {
        "model": model,
        "instructions": instructions or "You are a helpful assistant.",
        "input": input_items,
    }
    body = json.dumps(payload).encode("utf-8")
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if extra_headers:
        for k, v in extra_headers.items():
            if k and v is not None:
                headers[str(k)] = str(v)

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as e:
        raw_err = ""
        try:
            raw_err = e.read().decode("utf-8")
        except Exception:
            pass
        raise CodexCompactionError(
            f"Codex /responses/compact returned HTTP {e.code}",
            status=e.code,
            body=raw_err[:500] if raw_err else None,
        ) from e
    except urllib.error.URLError as e:
        raise CodexCompactionError(f"Codex /responses/compact transport error: {e}") from e

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise CodexCompactionError(
            f"Codex /responses/compact returned non-JSON body: {e}",
            status=status,
            body=raw[:500],
        ) from e

    if not isinstance(parsed, dict):
        raise CodexCompactionError(
            "Codex /responses/compact returned a non-object payload",
            status=status,
            body=raw[:500],
        )

    output = parsed.get("output")
    if not isinstance(output, list):
        raise CodexCompactionError(
            "Codex /responses/compact response missing 'output' list",
            status=status,
            body=raw[:500],
        )

    has_summary = any(
        isinstance(item, dict) and item.get("type") == "compaction"
        for item in output
    )
    if not has_summary:
        # Server returning no compaction means the input was already
        # short enough that nothing was compacted.  Surface this clearly so
        # the caller can fall back instead of silently shrinking the message
        # list to just the kept user turns (which would lose conversation).
        raise CodexNoCompactionError(
            "Codex /responses/compact returned no compaction item — "
            "input may have been below the server's compaction threshold",
            status=status,
        )

    cleaned_output = _strip_ids(output)
    usage = parsed.get("usage")
    if not isinstance(usage, dict):
        usage = {}

    logger.info(
        "Codex compaction: %d input items -> %d output items "
        "(input_tokens=%s, output_tokens=%s, total=%s)",
        len(input_items),
        len(cleaned_output),
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        usage.get("total_tokens"),
    )
    return {"output": cleaned_output, "usage": usage, "raw": parsed}


def build_compaction_envelope(
    *,
    output_items: List[Dict[str, Any]],
    n_compressed: int,
) -> Dict[str, Any]:
    """Wrap compaction output as a chat-style message that the Codex transport
    will splice back into the Responses input.

    The Codex transport recognises ``_codex_responses_items`` and emits each
    item verbatim (see ``agent.codex_responses_adapter._chat_messages_to_responses_input``).
    Other transports treat it as a normal user message and only see the
    display-only ``content`` text — they cannot decrypt the summary, so a
    session compacted via Codex is effectively pinned to the Codex backend
    until the next session reset.
    """
    label = (
        f"[Codex compaction: {n_compressed} prior turn(s) compressed into "
        f"an opaque server-side summary; only the Codex backend can decrypt it.]"
    )
    return {
        "role": "user",
        "content": label,
        "_codex_responses_items": output_items,
    }
