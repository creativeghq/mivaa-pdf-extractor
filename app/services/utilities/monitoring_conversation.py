"""The agent conversation a monitored subject reports into."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

MONITORING_AGENT_ID = "kai"


def agent_hub_path() -> str:
    return "/agent-hub?" + urlencode({"agent": MONITORING_AGENT_ID})


def alert_path(conversation_id: str, kind: str, tracked_id: str) -> str:
    """`alert` lets the page open a replacement chat if this one has since been deleted."""
    return "/agent-hub?" + urlencode({
        "agent": MONITORING_AGENT_ID,
        "conversation": conversation_id,
        "alert": f"{kind}:{tracked_id}",
    })


def absolute_url(url: str) -> str:
    """An email reader is not in the app, so a path is a dead link there."""
    if url.startswith(("http://", "https://")):
        return url
    base = (os.getenv("PUBLIC_APP_URL") or "https://app.materialshub.gr").rstrip("/")
    return f"{base}{url}"


def ensure_conversation(
    client: Any,
    *,
    table: str,
    tracked_id: str,
    user_id: str,
    title_prefix: str,
    label_columns: Sequence[str],
) -> Optional[str]:
    """The subject's conversation. Deleting a chat nulls the link, so the next alert opens a new one."""
    try:
        res = (
            client.table(table)
            .select(",".join(["source_conversation_id", *label_columns]))
            .eq("id", tracked_id)
            .limit(1)
            .execute()
        )
        row: Dict[str, Any] = (res.data or [{}])[0] or {}
        if row.get("source_conversation_id"):
            return row["source_conversation_id"]
        label = next((row[c] for c in label_columns if row.get(c)), "your subject")
        created = (
            client.table("agent_chat_conversations")
            .insert({"user_id": user_id, "agent_id": MONITORING_AGENT_ID, "title": f"{title_prefix} — {label}"})
            .execute()
        )
        conversation_id = ((created.data or [{}])[0] or {}).get("id")
        if not conversation_id:
            return None
        client.table(table).update({"source_conversation_id": conversation_id}).eq("id", tracked_id).execute()
        return conversation_id
    except Exception as e:
        logger.warning(f"monitoring-conversation: could not open one for {table}/{tracked_id}: {e}")
        return None


def post_alert(
    client: Any,
    *,
    conversation_id: str,
    title: str,
    body: str,
    kind: str,
    tracked_id: str,
    source_url: Optional[str] = None,
) -> bool:
    text = f"**{title}**"
    if body:
        text += f"\n\n{body}"
    if source_url and source_url.startswith(("http://", "https://")):
        text += f"\n\n[Open source]({source_url})"
    try:
        client.table("agent_chat_messages").insert({
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": text,
            "metadata": {
                "agentId": MONITORING_AGENT_ID,
                "chunk_type": "monitoring_alert",
                "tracked_kind": kind,
                "tracked_id": tracked_id,
            },
        }).execute()
    except Exception as e:
        logger.warning(f"monitoring-conversation: post failed (convo={conversation_id}): {e}")
        return False
    try:
        cur = (
            client.table("agent_chat_conversations")
            .select("message_count")
            .eq("id", conversation_id)
            .limit(1)
            .execute()
        )
        count = ((cur.data or [{}])[0] or {}).get("message_count") or 0
        now = datetime.now(timezone.utc).isoformat()
        client.table("agent_chat_conversations").update({
            "message_count": int(count) + 1,
            "last_message_at": now,
            "updated_at": now,
        }).eq("id", conversation_id).execute()
    except Exception as e:
        logger.debug(f"monitoring-conversation: bump failed (convo={conversation_id}): {e}")
    return True
