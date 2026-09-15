"""Append-only run log.

The loop records; it does not render. Everything the agent emits lands in one NDJSON
file, and any number of readers (`kopt watch`, `tail -f`, a later analysis script) consume
it independently. Nothing in the loop knows a viewer exists.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def runs_dir(project: Path) -> Path:
    return Path(project) / ".kopt" / "runs"


def new_run_log(project: Path) -> Path:
    """A fresh file per run, so finished runs stay inspectable."""
    return runs_dir(project) / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"


def list_run_logs(project: Path) -> list[Path]:
    d = runs_dir(project)
    return sorted(d.glob("*.jsonl")) if d.is_dir() else []


def latest_run_log(project: Path) -> Path | None:
    runs = list_run_logs(project)
    return runs[-1] if runs else None


class Recorder:
    """Writes one JSON object per line. Never raises into the loop."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()

    def write(self, kind: str, **fields) -> None:
        record = {"t": time.time(), "kind": kind, **fields}
        try:
            with self.path.open("a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass  # observability must never take down a run

    def attach(self, client) -> None:
        """Mirror every agent event into the log. `client` is a `kopt.agent.AgentBackend`;
        events arrive as plain dicts with a `type` key."""
        client.on_event(
            lambda event: self.write("event", type=event.get("type", "?"), data=event)
        )
