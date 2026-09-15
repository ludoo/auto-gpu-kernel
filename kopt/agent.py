"""Agent backends: the five things the loop needs from a coding agent.

The loop does not care which agent runs the iteration; it needs to open a session, send
one prompt and wait for the turn to settle, read cumulative stats to diff spend, know the
session id and model for the log, and mirror events into the run log. Everything else
(stopping rules, recovery turns, progress tracking) lives in `kopt.loop`.

Two backends:

- `OmpBackend` wraps `omp_rpc.RpcClient` (oh-my-pi). Needs the `[agent]` extra.
- `PiBackend` speaks pi's RPC protocol directly over stdin/stdout with the stdlib. Needs
  only a `pi` binary on PATH.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

EventListener = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class AgentState:
    session_id: str
    model_id: str


@dataclass(frozen=True)
class AgentStats:
    """Cumulative for the session; the loop diffs consecutive readings."""

    cost: float
    tokens: int
    tool_calls: int


@dataclass(frozen=True)
class Turn:
    assistant_text: str


class AgentBackend(Protocol):
    def start(self) -> "AgentBackend": ...
    def stop(self) -> None: ...
    def on_event(self, listener: EventListener) -> None:
        """Every event the agent emits, as a plain JSON-able dict with a `type` key."""
        ...
    def get_state(self) -> AgentState: ...
    def get_session_stats(self) -> AgentStats: ...
    def prompt_and_wait(self, prompt: str, timeout: float) -> Turn: ...


def plain(value: Any) -> Any:
    """Best-effort JSON-able view of an event (dataclass, dict, or scalar)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: plain(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# --- omp ---------------------------------------------------------------------


@dataclass
class OmpBackend:
    cwd: str
    model: str | None = None
    thinking: str | None = None
    max_time: str | None = None
    """Per-session wall clock passed to omp, e.g. "20m"."""
    request_timeout: float = 3600.0
    _client: Any = field(default=None, init=False, repr=False)

    def start(self) -> "OmpBackend":
        from omp_rpc import RpcClient

        extra = ["--max-time", self.max_time] if self.max_time else []
        self._client = RpcClient(
            cwd=self.cwd,
            model=self.model,
            thinking=self.thinking,
            extra_args=tuple(extra),
            request_timeout=self.request_timeout,
            # The default 10k-event ring is smaller than one verbose turn
            # (high-thinking iterations stream 10M+ tokens); overflow makes
            # prompt_and_wait lose agent_end and raises RpcError mid-run.
            max_event_history=None,
        ).start()
        self._client.install_headless_ui()
        return self

    def stop(self) -> None:
        if self._client is not None:
            self._client.stop()
            self._client = None

    def on_event(self, listener: EventListener) -> None:
        def forward(event: Any) -> None:
            data = plain(event)
            if not isinstance(data, dict):
                data = {"data": data}
            data.setdefault("type", getattr(event, "type", type(event).__name__))
            listener(data)

        self._client.on_event(forward)

    def get_state(self) -> AgentState:
        s = self._client.get_state()
        return AgentState(session_id=s.session_id, model_id=s.model.id)

    def get_session_stats(self) -> AgentStats:
        s = self._client.get_session_stats()
        return AgentStats(cost=s.cost, tokens=s.tokens.total, tool_calls=s.tool_calls)

    def prompt_and_wait(self, prompt: str, timeout: float) -> Turn:
        t = self._client.prompt_and_wait(prompt, timeout=timeout)
        return Turn(assistant_text=t.assistant_text or "")


# --- pi ----------------------------------------------------------------------


class PiError(RuntimeError):
    pass


@dataclass
class PiBackend:
    """pi in `--mode rpc`: JSON lines on stdin/stdout.

    Extension UI requests are answered so a headless run never blocks: `confirm` gets
    `false`, `select`/`input`/`editor` are cancelled, passive methods are ignored.
    """

    cwd: str
    model: str | None = None
    thinking: str | None = None
    extensions: tuple[str, ...] = ()
    """Extension paths to load. Discovery is disabled (`-ne`), so this list is the
    complete set; an empty tuple means built-in tools only."""
    exclude_tools: tuple[str, ...] = ()
    """Tool names to disable (`--exclude-tools`), the pi form of omp's deny list."""
    extra_args: tuple[str, ...] = ()
    binary: str = "pi"
    request_timeout: float = 3600.0
    _proc: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _reader: threading.Thread | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _cv: threading.Condition = field(default_factory=threading.Condition, init=False, repr=False)
    _responses: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _settled: int = field(default=0, init=False, repr=False)
    """Count of `agent_settled` events seen; a prompt waits for the count to advance."""
    _ids: Any = field(default_factory=lambda: itertools.count(1), init=False, repr=False)
    _listeners: list[EventListener] = field(default_factory=list, init=False, repr=False)
    _exited: bool = field(default=False, init=False, repr=False)
    _stderr: list[str] = field(default_factory=list, init=False, repr=False)
    """Last lines pi wrote to stderr, for the error message when it dies."""

    # --- process ---------------------------------------------------------
    def _argv(self) -> list[str]:
        # Discovery off for extensions and skills: the project's `.pi/skills` and the
        # configured extension list are the complete surface, whatever the user's
        # global pi setup looks like. `--approve` trusts the project-local `.pi/`.
        argv = [
            self.binary, "--mode", "rpc", "--approve",
            "--no-extensions", "--no-skills", "--skill", ".pi/skills",
        ]
        if self.exclude_tools:
            argv += ["--exclude-tools", ",".join(self.exclude_tools)]
        if self.model:
            argv += ["--model", self.model]
        if self.thinking:
            argv += ["--thinking", self.thinking]
        for ext in self.extensions:
            argv += ["--extension", os.path.expanduser(ext)]
        argv += list(self.extra_args)
        return argv

    def start(self) -> "PiBackend":
        if shutil.which(self.binary) is None:
            raise PiError(f"{self.binary!r} not found on PATH")
        self._proc = subprocess.Popen(
            self._argv(),
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        threading.Thread(target=self._read_stderr, daemon=True).start()
        return self

    def _read_stderr(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stderr is not None
        for line in proc.stderr:
            if line.startswith("Warning: No models match"):
                continue  # stale enabledModels patterns, noise for every run
            self._stderr.append(line.rstrip())
            del self._stderr[:-20]

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    # --- wire ------------------------------------------------------------
    def _read(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._dispatch(msg)
        with self._cv:
            self._exited = True
            self._cv.notify_all()

    def _dispatch(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "response":
            with self._cv:
                self._responses[msg.get("id", "")] = msg
                self._cv.notify_all()
            return
        if kind == "extension_ui_request":
            self._answer_ui(msg)
        for fn in list(self._listeners):
            try:
                fn(msg)
            except Exception:
                pass  # observability never takes down a run
        if kind == "agent_settled":
            with self._cv:
                self._settled += 1
                self._cv.notify_all()

    def _answer_ui(self, req: dict) -> None:
        method = req.get("method")
        rid = req.get("id")
        if rid is None:
            return
        if method == "confirm":
            self._send({"type": "extension_ui_response", "id": rid, "confirmed": False})
        elif method in ("select", "input", "editor"):
            self._send({"type": "extension_ui_response", "id": rid, "cancelled": True})
        # notify, setStatus, setWidget, setTitle, set_editor_text: fire-and-forget

    def _stderr_tail(self) -> str:
        return ("\n  stderr: " + "\n  ".join(self._stderr)) if self._stderr else ""

    def _send(self, msg: dict) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise PiError("pi is not running")
        with self._lock:
            try:
                proc.stdin.write(json.dumps(msg) + "\n")
                proc.stdin.flush()
            except (BrokenPipeError, ValueError) as exc:
                raise PiError("pi exited") from exc

    def _request(self, msg: dict, timeout: float | None = None) -> dict:
        rid = f"kopt-{next(self._ids)}"
        self._send({"id": rid, **msg})
        deadline_timeout = self.request_timeout if timeout is None else timeout
        with self._cv:
            self._cv.wait_for(
                lambda: rid in self._responses or self._exited, timeout=deadline_timeout
            )
            if rid in self._responses:
                resp = self._responses.pop(rid)
            elif self._exited:
                raise PiError("pi exited before responding" + self._stderr_tail())
            else:
                raise PiError(f"timeout waiting for {msg.get('type')}")
        if not resp.get("success", False):
            raise PiError(f"{msg.get('type')}: {resp.get('error', 'failed')}")
        return resp.get("data") or {}

    # --- AgentBackend ----------------------------------------------------
    def on_event(self, listener: EventListener) -> None:
        self._listeners.append(listener)

    def get_state(self) -> AgentState:
        d = self._request({"type": "get_state"})
        model = d.get("model") or {}
        return AgentState(session_id=d.get("sessionId", ""), model_id=model.get("id", ""))

    def get_session_stats(self) -> AgentStats:
        d = self._request({"type": "get_session_stats"})
        return AgentStats(
            cost=float(d.get("cost", 0.0)),
            tokens=int((d.get("tokens") or {}).get("total", 0)),
            tool_calls=int(d.get("toolCalls", 0)),
        )

    def prompt_and_wait(self, prompt: str, timeout: float) -> Turn:
        with self._cv:
            target = self._settled + 1
        self._request({"type": "prompt", "message": prompt}, timeout=60)
        with self._cv:
            ok = self._cv.wait_for(lambda: self._settled >= target or self._exited, timeout=timeout)
            if self._exited and self._settled < target:
                raise PiError("pi exited mid-turn" + self._stderr_tail())
            if not ok:
                raise PiError(f"turn did not settle within {timeout:.0f}s")
        d = self._request({"type": "get_last_assistant_text"})
        return Turn(assistant_text=d.get("text") or "")


# --- config ------------------------------------------------------------------

BACKENDS = ("omp", "pi")


@dataclass(frozen=True)
class AgentConfig:
    kind: str = "omp"
    extensions: tuple[str, ...] = ()
    exclude_tools: tuple[str, ...] = ()


def load_agent_config(project: "os.PathLike[str] | str") -> AgentConfig:
    """The `[agent]` table of the project's config.toml; omp when absent, so projects
    scaffolded before the table existed keep working."""
    import tomllib
    from pathlib import Path

    path = Path(project) / "config.toml"
    try:
        raw = tomllib.loads(path.read_text()).get("agent") or {}
    except (OSError, tomllib.TOMLDecodeError):
        return AgentConfig()
    kind = raw.get("kind", "omp")
    if kind not in BACKENDS:
        raise SystemExit(f"config.toml: agent.kind {kind!r} not in {', '.join(BACKENDS)}")
    return AgentConfig(
        kind=kind,
        extensions=tuple(str(e) for e in raw.get("extensions", [])),
        exclude_tools=tuple(str(t) for t in raw.get("exclude_tools", [])),
    )


# --- factory -----------------------------------------------------------------


def make_backend(
    kind: str,
    *,
    cwd: str,
    model: str | None,
    thinking: str | None,
    request_timeout: float,
    max_time: str | None = None,
    extensions: tuple[str, ...] = (),
    exclude_tools: tuple[str, ...] = (),
    extra_args: tuple[str, ...] = (),
) -> AgentBackend:
    if kind == "omp":
        return OmpBackend(
            cwd=cwd, model=model, thinking=thinking, max_time=max_time,
            request_timeout=request_timeout,
        )
    if kind == "pi":
        return PiBackend(
            cwd=cwd, model=model, thinking=thinking, extensions=extensions,
            exclude_tools=exclude_tools, extra_args=extra_args,
            request_timeout=request_timeout,
        )
    raise SystemExit(f"unknown agent backend {kind!r}; have: {', '.join(BACKENDS)}")
