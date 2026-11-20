"""Worker messaging: spawn / send / inbox, delivery rules, durability, launcher.

Delivery tests drive the real FastAPI app with a real ``RunManager`` and a
fake process factory (no agent CLI is launched), through
``httpx.ASGITransport`` so orchestrator tasks share one event loop.
"""

from __future__ import annotations

import asyncio
import io
import json
import socket
import subprocess
import sys
import threading
import time

import httpx
import pytest
import uvicorn

from agent_harness import api as api_module
from agent_harness import worker_cli
from agent_harness.api import create_app
from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.messaging import format_delivery_turn, self_address
from agent_harness.models import MailMessage, Project, Session, parse_address
from agent_harness.orchestrator import (
    ClaudeCodeCommandBuilder,
    PiCommandBuilder,
    ProcessCommand,
    RunManager,
)
from agent_harness.repository import InMemoryRepository
from agent_harness.storage import open_sqlite_repository

PROJECT = {"path": "/tmp/p", "name": "p"}


class _Stream:
    async def readline(self) -> bytes:
        return b""


class FakeProcess:
    """A CLI process that runs until ``finish()``; thread-safe to finish."""

    def __init__(self, command: ProcessCommand) -> None:
        self.command = command
        self.stdout = _Stream()
        self.stderr = _Stream()
        self.returncode: int | None = None
        self.pid = 0
        self._done = threading.Event()

    async def wait(self) -> int:
        while not self._done.is_set():
            await asyncio.sleep(0.005)
        return self.returncode or 0

    def finish(self, code: int = 0) -> None:
        self.returncode = code
        self._done.set()

    def terminate(self) -> None:
        self.finish(-15)

    @property
    def text(self) -> str:
        return self.command.argv[-1]


class Harness:
    def __init__(self, repo=None, *, execute=True, builders=None, self_url="http://127.0.0.1:9999"):
        self.repo = repo or InMemoryRepository()
        self.bus = DurableEventBus(self.repo) if hasattr(self.repo, "append_event") else InMemoryEventBus()
        self.spawned: list[FakeProcess] = []

        async def factory(command: ProcessCommand) -> FakeProcess:
            proc = FakeProcess(command)
            self.spawned.append(proc)
            return proc

        self.manager = RunManager(event_bus=self.bus, process_factory=factory) if execute else None
        self.app = create_app(
            repository=self.repo,
            event_bus=self.bus,
            run_manager=self.manager,
            command_builders=builders or {"claude-code": ClaudeCodeCommandBuilder(), "pi": PiCommandBuilder()},
            self_url=self_url,
        )


async def _settle() -> None:
    for _ in range(40):
        await asyncio.sleep(0.005)


async def _new_session(client: httpx.AsyncClient, backend: str = "claude-code") -> str:
    resp = await client.post("/v1/sessions", json={"backend": backend, "project": PROJECT})
    assert resp.status_code == 201
    return resp.json()["id"]


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture(autouse=True)
def _claude_transcript_follows_runs(monkeypatch):
    # The claude builder resumes once the transcript exists on disk. Model
    # that as "a run has been started for this session before".
    started: set[str] = set()
    real_submit = RunManager.submit

    def submit(self, *, session, run, command, on_start=None):
        started.add(session.id)
        return real_submit(self, session=session, run=run, command=command, on_start=on_start)

    monkeypatch.setattr(RunManager, "submit", submit)
    monkeypatch.setattr(
        api_module, "claude_conversation_exists", lambda session: session.id in started
    )


# --- turn-based delivery ------------------------------------------------------


@pytest.mark.asyncio
async def test_send_to_idle_worker_resumes_same_conversation_in_new_process() -> None:
    h = Harness()
    async with _client(h.app) as client:
        worker = await _new_session(client)

        first = await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "brief", "sender": "ses_boss"})
        assert first.status_code == 202
        assert first.json()["status"] == "delivered"
        await _settle()
        assert len(h.spawned) == 1
        h.spawned[0].finish()
        await _settle()

        reply = await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "LGTM", "sender": "ses_rev"})
        body = reply.json()
        assert body["status"] == "delivered"
        await _settle()

    assert len(h.spawned) == 2
    uuid = worker.removeprefix("ses_")
    first_argv, second_argv = h.spawned[0].command.argv, h.spawned[1].command.argv
    assert "--session-id" in first_argv
    # Same conversation, new process: --resume of the same claude UUID.
    resume_at = second_argv.index("--resume")
    assert second_argv[resume_at + 1].replace("-", "") == uuid
    assert "LGTM" in h.spawned[1].text and "from ses_rev" in h.spawned[1].text
    assert "agent-harness worker send ses_rev" in h.spawned[1].text
    env = h.spawned[1].command.env
    assert env["AGENT_HARNESS_SESSION"] == worker
    assert env["AGENT_HARNESS_URL"] == "http://127.0.0.1:9999"

    mail = h.repo.list_mail(worker)
    assert [m.state for m in mail] == ["delivered", "delivered"]
    assert mail[1].run_id == body["run_id"]


@pytest.mark.asyncio
async def test_pi_worker_resumes_with_same_session_id() -> None:
    h = Harness()
    async with _client(h.app) as client:
        worker = await _new_session(client, "pi")
        await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "one"})
        await _settle()
        h.spawned[0].finish()
        await _settle()
        await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "two"})
        await _settle()
    ids = [p.command.argv[p.command.argv.index("--session-id") + 1] for p in h.spawned]
    assert len(ids) == 2 and ids[0] == ids[1]
    assert "anonymous" in h.spawned[1].text


@pytest.mark.asyncio
async def test_send_to_busy_worker_waits_and_is_delivered_after_the_run() -> None:
    h = Harness()
    async with _client(h.app) as client:
        worker = await _new_session(client)
        await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "brief"})
        await _settle()
        assert len(h.spawned) == 1

        a = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "msg A", "sender": "ses_x"})).json()
        b = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "msg B", "sender": "ses_y"})).json()
        assert a["status"] == b["status"] == "waiting"
        status = (await client.get("/v1/mailbox/status")).json()
        assert status == {"pending": 2, "active_runs": 1, "executes_runs": True, "loopback_only": False}
        await _settle()
        assert len(h.spawned) == 1  # nothing started while busy

        h.spawned[0].finish()
        await _settle()
        # Both waiting messages go to ONE new turn, in order.
        assert len(h.spawned) == 2
        text = h.spawned[1].text
        assert text.index("msg A") < text.index("msg B")
        assert "2 messages" in text

        h.spawned[1].finish()
        await _settle()
        assert len(h.spawned) == 2  # delivered at most once
        assert (await client.get("/v1/mailbox/status")).json()["pending"] == 0


@pytest.mark.asyncio
async def test_mail_waits_behind_a_chat_run_queue() -> None:
    # A busy session may also have ordinary queued runs (e.g. from a chat
    # bridge); mail is delivered only once all of them are done.
    h = Harness()
    async with _client(h.app) as client:
        worker = await _new_session(client)
        await client.post(f"/v1/sessions/{worker}/runs", json={"message": "chat 1"})
        await client.post(f"/v1/sessions/{worker}/runs", json={"message": "chat 2"})
        sent = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "mail"})).json()
        assert sent["status"] == "waiting"
        await _settle()
        h.spawned[0].finish()
        await _settle()
        assert h.spawned[1].text == "chat 2"
        h.spawned[1].finish()
        await _settle()
        assert len(h.spawned) == 3 and "mail" in h.spawned[2].text


# --- interactive (external) sessions -----------------------------------------


def _external_session(repo) -> Session:
    session = Session(
        id="ses_" + "a" * 32, backend="claude-code", project=Project(**PROJECT), origin="external"
    )
    repo.upsert_session(session)
    return session


@pytest.mark.asyncio
async def test_interactive_session_mail_is_stored_never_resumed() -> None:
    h = Harness()
    tui = _external_session(h.repo)
    async with _client(h.app) as client:
        resp = await client.post(f"/v1/sessions/{tui.id}/mailbox", json={"body": "review done", "sender": "ses_rev"})
        assert resp.status_code == 202
        assert resp.json()["status"] == "stored"
        # Even a later run end elsewhere / status poll never starts a turn.
        other = await _new_session(client)
        await client.post(f"/v1/sessions/{other}/mailbox", json={"body": "x"})
        await _settle()
        h.spawned[0].finish()
        await _settle()
        assert all(p.command.env["AGENT_HARNESS_SESSION"] == other for p in h.spawned)
        assert (await client.get("/v1/mailbox/status")).json()["pending"] == 0

        unread = (await client.get(f"/v1/sessions/{tui.id}/mailbox", params={"state": "unread"})).json()["data"]
        assert [m["body"] for m in unread] == ["review done"]
        acked = (await client.post(f"/v1/sessions/{tui.id}/mailbox/ack", json={"ids": [unread[0]["id"]]})).json()
        assert acked["data"][0]["state"] == "read"
        assert (await client.get(f"/v1/sessions/{tui.id}/mailbox", params={"state": "unread"})).json()["data"] == []


@pytest.mark.asyncio
async def test_unknown_and_remote_addresses_are_refused() -> None:
    h = Harness()
    async with _client(h.app) as client:
        assert (await client.post("/v1/sessions/ses_nope/mailbox", json={"body": "x"})).status_code == 404
        remote = await client.post("/v1/sessions/ses_a@boxd-1/mailbox", json={"body": "x"})
        assert remote.status_code == 422
        assert "not supported yet" in remote.json()["detail"]
    with pytest.raises(ValueError):
        parse_address("ses_a@host")


@pytest.mark.asyncio
async def test_without_execute_runs_mail_stays_pending() -> None:
    h = Harness(execute=False)
    async with _client(h.app) as client:
        worker = await _new_session(client)
        sent = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "x"})).json()
        assert sent["status"] == "waiting"
        assert sent["message"]["state"] == "pending"


# --- durability ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_pending_mail_survives_restart_and_is_delivered_once(tmp_path) -> None:
    db = tmp_path / "h.db"
    repo1 = open_sqlite_repository(db)
    first = Harness(repo1)
    async with first.app.router.lifespan_context(first.app), _client(first.app) as client:
        worker = await _new_session(client)
        await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "brief"})
        await _settle()
        waiting = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "after restart"})).json()
        assert waiting["status"] == "waiting"
    # Harness dies with the brief's run unfinished.
    first.spawned[0].finish(-9)
    repo1.close()

    repo2 = open_sqlite_repository(db)
    second = Harness(repo2)
    async with second.app.router.lifespan_context(second.app):
        await _settle()
        assert len(second.spawned) == 1
        assert "after restart" in second.spawned[0].text
        assert "brief" not in second.spawned[0].text  # already delivered once
        second.spawned[0].finish()
        await _settle()
    repo2.close()

    repo3 = open_sqlite_repository(db)
    third = Harness(repo3)
    async with third.app.router.lifespan_context(third.app):
        await _settle()
        assert third.spawned == []
        states = [m.state for m in repo3.list_mail(worker)]
        assert states == ["delivered", "delivered"]
    repo3.close()


def test_sqlite_claims_are_never_released_by_a_restart(tmp_path) -> None:
    repo = open_sqlite_repository(tmp_path / "h.db")
    tui = _external_session(repo)
    repo.add_mail(MailMessage(to=tui.id, body="hi", state="unread", delivery="inbox"))
    harness_session = Session(backend="pi", project=Project(**PROJECT))
    repo.upsert_session(harness_session)
    repo.add_mail(MailMessage(to=harness_session.id, body="work"))
    assert len(repo.claim_pending_mail(harness_session.id)) == 1
    assert repo.claim_pending_mail(harness_session.id) == []
    repo.close()

    repo = open_sqlite_repository(tmp_path / "h.db")
    assert [m.body for m in repo.list_mail(tui.id, {"unread"})] == ["hi"]
    # Claimed without a run link (e.g. the link write failed): still claimed.
    assert repo.sessions_with_pending_mail() == []
    assert repo.claim_pending_mail(harness_session.id) == []
    repo.close()


def test_two_connections_cannot_both_claim_the_same_mail(tmp_path) -> None:
    # Both connections read the pending row before either writes, as two
    # harness processes on one database could; only one claim may win.
    db = tmp_path / "h.db"
    first, second = open_sqlite_repository(db), open_sqlite_repository(db)
    session = Session(backend="pi", project=Project(**PROJECT))
    first.upsert_session(session)
    first.add_mail(MailMessage(to=session.id, body="once"))
    barrier = threading.Barrier(2, timeout=5)

    class _BarrierConnection:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *params):
            if sql.lstrip().startswith("update mailbox"):
                barrier.wait()
            return self._inner.execute(sql, *params)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    for repo in (first, second):
        repo._connection = _BarrierConnection(repo._connection)
    results: dict[str, list] = {}

    def claim(name, repo):
        results[name] = repo.claim_pending_mail(session.id)

    threads = [threading.Thread(target=claim, args=(n, r)) for n, r in (("a", first), ("b", second))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert sorted(len(v) for v in results.values()) == [0, 1]


@pytest.mark.asyncio
async def test_failed_run_link_write_never_causes_a_second_delivery(tmp_path, monkeypatch) -> None:
    db = tmp_path / "h.db"
    repo1 = open_sqlite_repository(db)

    def broken_attach(ids, run_id):
        raise OSError("disk full")

    monkeypatch.setattr(repo1, "attach_mail_run", broken_attach)
    first = Harness(repo1)
    async with first.app.router.lifespan_context(first.app), _client(first.app) as client:
        worker = await _new_session(client)
        sent = await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "only once"})
        assert sent.status_code == 202
        await _settle()
        assert len(first.spawned) == 1
        first.spawned[0].finish()
        await _settle()
    repo1.close()

    repo2 = open_sqlite_repository(db)
    second = Harness(repo2)
    async with second.app.router.lifespan_context(second.app):
        await _settle()
        assert second.spawned == []
    repo2.close()


@pytest.mark.asyncio
async def test_large_batches_are_split_to_fit_one_cli_argument() -> None:
    h = Harness()
    async with _client(h.app) as client:
        worker = await _new_session(client)
        await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "brief"})
        await _settle()
        for i in range(4):
            body = str(i) * 30_000
            resp = await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": body})
            assert resp.json()["status"] == "waiting"
        h.spawned[0].finish()
        await _settle()
        assert len(h.spawned) == 2
        assert len(h.spawned[1].text.encode()) < 128 * 1024
        assert "3 messages" in h.spawned[1].text
        h.spawned[1].finish()
        await _settle()
        assert len(h.spawned) == 3 and "3" * 30_000 in h.spawned[2].text


@pytest.mark.asyncio
async def test_bodies_that_cannot_be_a_cli_argument_are_refused() -> None:
    h = Harness()
    async with _client(h.app) as client:
        worker = await _new_session(client)
        nul = await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "a\x00b"})
        big = await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "x" * 30_001})
        assert nul.status_code == big.status_code == 422
        assert h.repo.list_mail(worker) == []


@pytest.mark.asyncio
async def test_codex_mail_waits_until_its_resume_id_is_bound() -> None:
    from agent_harness.orchestrator import CodexCommandBuilder

    h = Harness(builders={"codex": CodexCommandBuilder()})
    async with _client(h.app) as client:
        worker = await _new_session(client, "codex")
        await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "brief"})
        await _settle()
        h.spawned[0].finish()
        await _settle()
        # No observer bound the codex thread: a fresh ``codex exec`` would
        # lose the conversation, so the follow-up waits.
        sent = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "follow-up"})).json()
        assert sent["status"] == "waiting" and len(h.spawned) == 1

        session = h.repo.get_session(worker)
        h.repo.upsert_session(session.model_copy(update={"codex_resume_id": "0199aaaa-bbbb-7ccc-8ddd-eeeeffff0000"}))
        resent = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "again"})).json()
        assert resent["status"] == "delivered"
        await _settle()
    argv = h.spawned[1].command.argv
    assert argv[:3] == ("codex", "exec", "resume") and "0199aaaa-bbbb-7ccc-8ddd-eeeeffff0000" in argv
    assert "follow-up" in h.spawned[1].text and "again" in h.spawned[1].text


@pytest.mark.asyncio
async def test_storage_failure_before_submission_returns_mail_to_pending(monkeypatch) -> None:
    h = Harness()
    async with _client(h.app) as client:
        worker = await _new_session(client, "pi")
        await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "brief"})
        await _settle()
        waiting = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "reply"})).json()
        assert waiting["status"] == "waiting"

        real_create_run = h.repo.create_run
        calls = {"n": 0}

        def flaky_create_run(session_id, request):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("disk I/O error")
            return real_create_run(session_id, request)

        monkeypatch.setattr(h.repo, "create_run", flaky_create_run)
        h.spawned[0].finish()
        await _settle()
        assert len(h.spawned) == 1  # the delivery run could not be created
        assert [m.state for m in h.repo.list_mail(worker)] == ["delivered", "pending"]

        # The next trigger delivers it, exactly once.
        again = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "nudge"})).json()
        assert again["status"] == "delivered"
        await _settle()
        assert len(h.spawned) == 2 and "reply" in h.spawned[1].text and "nudge" in h.spawned[1].text


@pytest.mark.asyncio
async def test_send_reports_delivered_even_if_the_run_link_write_fails(monkeypatch) -> None:
    h = Harness()

    def broken_attach(ids, run_id):
        raise OSError("disk full")

    monkeypatch.setattr(h.repo, "attach_mail_run", broken_attach)
    async with _client(h.app) as client:
        worker = await _new_session(client, "pi")
        sent = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "x"})).json()
        assert sent["status"] == "delivered" and sent["run_id"]


@pytest.mark.asyncio
async def test_late_codex_binding_alone_delivers_the_waiting_reply() -> None:
    from agent_harness.models import Event
    from agent_harness.orchestrator import CodexCommandBuilder

    h = Harness(builders={"codex": CodexCommandBuilder()})
    async with h.app.router.lifespan_context(h.app), _client(h.app) as client:
        worker = await _new_session(client, "codex")
        await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "brief"})
        await _settle()
        h.spawned[0].finish()
        await _settle()
        sent = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "the reply"})).json()
        assert sent["status"] == "waiting"

        # What the observer does when it binds the rollout late.
        bound = h.repo.get_session(worker).model_copy(
            update={"codex_resume_id": "0199aaaa-bbbb-7ccc-8ddd-eeeeffff0000"}
        )
        h.repo.upsert_session(bound)
        await h.bus.publish(Event(event="session.updated", session_id=worker,
                                  data={"session": bound.model_dump(mode="json")}))
        for _ in range(100):
            if len(h.spawned) == 2:
                break
            await asyncio.sleep(0.01)
        assert len(h.spawned) == 2
        argv = h.spawned[1].command.argv
        assert argv[:3] == ("codex", "exec", "resume") and "the reply" in h.spawned[1].text


@pytest.mark.asyncio
async def test_rendered_batch_with_many_small_messages_fits_one_argument() -> None:
    h = Harness()
    async with _client(h.app) as client:
        worker = await _new_session(client, "pi")
        await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "brief"})
        await _settle()
        sender = "ses_" + "f" * 32
        for _ in range(400):
            await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "y" * 240, "sender": sender})
        for i in range(10):
            h.spawned[-1].finish()
            await _settle()
            if (await client.get("/v1/mailbox/status")).json()["pending"] == 0:
                break
        turns = h.spawned[1:]
        assert len(turns) >= 2
        assert all(len(p.text.encode()) < 128 * 1024 for p in turns)
        assert sum(p.text.count("--- message ") for p in turns) == 400


@pytest.mark.asyncio
async def test_failure_after_run_row_insert_does_not_block_the_first_codex_turn(monkeypatch) -> None:
    from agent_harness.orchestrator import CodexCommandBuilder

    h = Harness(builders={"codex": CodexCommandBuilder()})
    real_list_messages = h.repo.list_messages
    fail = {"once": True}

    def flaky_list_messages(session_id):
        if fail["once"]:
            fail["once"] = False
            raise OSError("disk I/O error")
        return real_list_messages(session_id)

    async with _client(h.app) as client:
        worker = await _new_session(client, "codex")
        monkeypatch.setattr(h.repo, "list_messages", flaky_list_messages)
        first = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "the brief"})).json()
        assert first["status"] == "waiting" and h.spawned == []
        # The unsubmitted run row is closed, not left queued.
        assert [r.status for r in h.repo.list_runs(worker)] == ["failed"]

        second = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "nudge"})).json()
        assert second["status"] == "delivered"
        await _settle()
    assert len(h.spawned) == 1
    argv = h.spawned[0].command.argv
    # Still the first turn: a fresh ``codex exec``, carrying the brief once.
    assert argv[:2] == ("codex", "exec") and argv[2] != "resume"
    assert h.spawned[0].text.count("the brief") == 1 and "nudge" in h.spawned[0].text


@pytest.mark.asyncio
async def test_binding_watcher_survives_a_transient_storage_error(monkeypatch) -> None:
    from agent_harness.models import Event
    from agent_harness.orchestrator import CodexCommandBuilder

    h = Harness(builders={"codex": CodexCommandBuilder()})
    async with h.app.router.lifespan_context(h.app), _client(h.app) as client:
        workers = []
        for _ in range(2):
            worker = await _new_session(client, "codex")
            await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": "brief"})
            await _settle()
            h.spawned[-1].finish()
            await _settle()
            sent = (await client.post(f"/v1/sessions/{worker}/mailbox", json={"body": f"reply {worker}"})).json()
            assert sent["status"] == "waiting"
            workers.append(worker)

        real_list_mail = h.repo.list_mail
        fail = {"once": True}

        def flaky_list_mail(session_id, states=None):
            if fail["once"]:
                fail["once"] = False
                raise OSError("database is locked")
            return real_list_mail(session_id, states)

        monkeypatch.setattr(h.repo, "list_mail", flaky_list_mail)

        async def bind(worker: str, uuid: str) -> None:
            bound = h.repo.get_session(worker).model_copy(update={"codex_resume_id": uuid})
            h.repo.upsert_session(bound)
            await h.bus.publish(Event(event="session.updated", session_id=worker,
                                      data={"session": bound.model_dump(mode="json")}))

        await bind(workers[0], "0199aaaa-bbbb-7ccc-8ddd-eeeeffff0001")
        for _ in range(200):
            if len(h.spawned) == 3:
                break
            await asyncio.sleep(0.01)
        assert len(h.spawned) == 3 and f"reply {workers[0]}" in h.spawned[2].text

        # The watcher is still alive for other sessions.
        await bind(workers[1], "0199aaaa-bbbb-7ccc-8ddd-eeeeffff0002")
        for _ in range(200):
            if len(h.spawned) == 4:
                break
            await asyncio.sleep(0.01)
        assert len(h.spawned) == 4 and f"reply {workers[1]}" in h.spawned[3].text


def test_launcher_refuses_reuse_of_a_harness_that_is_not_loopback_only(monkeypatch) -> None:
    def fake_request(url, method="GET", body=None, *, timeout=15.0):
        if url.endswith("/v1/backends"):
            return {"data": []}
        return {"pending": 0, "active_runs": 0, "executes_runs": True, "loopback_only": False}

    monkeypatch.setattr(worker_cli, "_request", fake_request)
    with pytest.raises(worker_cli.HarnessError, match="not bound to loopback"):
        worker_cli._probe("http://127.0.0.1:1")


def test_launcher_does_not_start_a_second_harness_when_the_port_is_slow(tmp_path, capsys) -> None:
    # A listener that accepts but never answers looks like a busy harness,
    # not an empty port: up must refuse instead of starting another one.
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(8)
    port = silent.getsockname()[1]
    try:
        from agent_harness.cli import main

        assert main(["up", "--port", str(port), "--state-dir", str(tmp_path)]) == 1
        assert "could not verify" in capsys.readouterr().err
        assert not (tmp_path / f"launcher-{port}.json").exists()
    finally:
        silent.close()


# --- address helpers ------------------------------------------------------------


def test_self_address_precedence() -> None:
    claude_uuid = "0f8b3c1e-1111-2222-3333-444455556666"
    assert self_address({"AGENT_HARNESS_SESSION": "ses_w", "CLAUDE_CODE_SESSION_ID": claude_uuid}) == "ses_w"
    assert self_address({"CLAUDE_CODE_SESSION_ID": claude_uuid}) == "ses_0f8b3c1e111122223333444455556666"
    assert self_address({"CLAUDE_CODE_SESSION_ID": "garbage"}) is None
    assert self_address({}) is None


def test_delivery_turn_without_sender_has_no_reply_hint() -> None:
    session = Session(id="ses_w", backend="pi", project=Project(**PROJECT))
    text = format_delivery_turn(session, [MailMessage(to="ses_w", body="hello")])
    assert "your address: ses_w" in text and "no reply address" in text


# --- CLI against a real HTTP server ------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_harness():
    port = _free_port()
    h = Harness(self_url=f"http://127.0.0.1:{port}")
    server = uvicorn.Server(uvicorn.Config(h.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    h.url = f"http://127.0.0.1:{port}"
    yield h
    for proc in h.spawned:
        proc.finish()
    server.should_exit = True
    thread.join(timeout=10)


def _wait_idle(url: str) -> None:
    for _ in range(500):
        if worker_cli._request(f"{url}/v1/mailbox/status")["active_runs"] == 0:
            return
        time.sleep(0.01)
    raise AssertionError("harness did not become idle")


def _cli(capsys, *argv: str) -> dict:
    from agent_harness.cli import main

    assert main(list(argv)) == 0
    return json.loads(capsys.readouterr().out)


def test_cli_spawn_returns_immediately_and_reply_resumes_spawner(live_harness, capsys, monkeypatch, tmp_path) -> None:
    h = live_harness
    monkeypatch.setenv("AGENT_HARNESS_URL", h.url)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    # The implementer is itself a turn-based worker (its run has ended).
    boss = _cli(capsys, "worker", "spawn", "--backend", "claude-code", "--cwd", str(tmp_path), "you lead")
    for _ in range(200):
        if h.spawned:
            break
        time.sleep(0.01)
    h.spawned[0].finish()
    _wait_idle(h.url)
    monkeypatch.setenv("AGENT_HARNESS_SESSION", boss["address"])

    started = time.monotonic()
    reviewer = _cli(capsys, "worker", "spawn", "--backend", "pi", "--model", "m1", "review the diff")
    assert time.monotonic() - started < 5
    assert reviewer["status"] == "delivered" and reviewer["reply_to"] == boss["address"]
    # The reviewer's CLI process is still running: spawn did not wait for it.
    assert h.spawned[-1].returncode is None
    assert "review the diff" in h.spawned[-1].text and f"from {boss['address']}" in h.spawned[-1].text

    # The reviewer replies using its own environment.
    monkeypatch.setenv("AGENT_HARNESS_SESSION", reviewer["address"])
    sent = _cli(capsys, "worker", "send", boss["address"], "looks good")
    # "waiting" only if the boss's finished run is still being recorded;
    # either way the reply must resume the boss exactly once.
    assert sent["status"] in ("delivered", "waiting")
    for _ in range(200):
        if len(h.spawned) == 3:
            break
        time.sleep(0.01)
    time.sleep(0.1)
    assert len(h.spawned) == 3
    resumed = h.spawned[2]
    assert resumed.command.env["AGENT_HARNESS_SESSION"] == boss["address"]
    assert "--resume" in resumed.command.argv and "looks good" in resumed.text


def test_cli_inbox_for_interactive_session(live_harness, capsys, monkeypatch) -> None:
    h = live_harness
    tui = _external_session(h.repo)
    monkeypatch.delenv("AGENT_HARNESS_SESSION", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "a" * 8 + "-" + "a" * 4 + "-" + "a" * 4 + "-" + "a" * 4 + "-" + "a" * 12)
    monkeypatch.setattr(sys, "stdin", io.StringIO("from stdin"))
    out = _cli(capsys, "worker", "send", "--url", h.url, "--as", "ses_rev", tui.id, "-")
    assert out["status"] == "stored"
    assert h.spawned == []

    from agent_harness.cli import main

    assert main(["worker", "inbox", "--url", h.url]) == 0
    text = capsys.readouterr().out
    assert "from ses_rev" in text and "from stdin" in text
    assert main(["worker", "inbox", "--url", h.url]) == 0
    assert "No messages" in capsys.readouterr().out
    assert main(["worker", "send", "--url", h.url, "ses_missing", "x"]) == 1
    assert "404" in capsys.readouterr().err


def test_cli_spawn_inherits_permission_bypass_from_its_spawner(live_harness, capsys, monkeypatch) -> None:
    h = live_harness
    monkeypatch.setenv("AGENT_HARNESS_URL", h.url)
    monkeypatch.delenv("AGENT_HARNESS_SESSION", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    # No known spawner: restricted unless asked.
    assert _cli(capsys, "worker", "spawn", "--backend", "pi", "x")["bypass_permissions"] is False
    open_parent = _cli(capsys, "worker", "spawn", "--backend", "pi", "--bypass-permissions", "x")
    assert open_parent["bypass_permissions"] is True

    monkeypatch.setenv("AGENT_HARNESS_SESSION", open_parent["address"])
    child = _cli(capsys, "worker", "spawn", "--backend", "codex", "x")
    assert child["bypass_permissions"] is True
    assert _cli(capsys, "worker", "spawn", "--backend", "codex", "--no-bypass-permissions", "x")["bypass_permissions"] is False

    restricted = _cli(capsys, "worker", "spawn", "--backend", "pi", "--no-bypass-permissions", "x")
    monkeypatch.setenv("AGENT_HARNESS_SESSION", restricted["address"])
    assert _cli(capsys, "worker", "spawn", "--backend", "codex", "x")["bypass_permissions"] is False


# --- launcher -------------------------------------------------------------------




def _cmdline(pid: int) -> list[str]:
    return [w.decode() for w in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if w]


def _mode(path) -> int:
    import os
    import stat

    return stat.S_IMODE(os.stat(path).st_mode)


def test_launcher_starts_reuses_and_stops_only_what_it_started(tmp_path, capsys) -> None:
    port = _free_port()
    state = tmp_path / "state"
    base = ["--port", str(port), "--state-dir", str(state)]
    # A forwarded --host must not override the launcher's loopback bind.
    up = _cli(capsys, "up", *base, "--serve-arg=--no-observer", "--serve-arg=--host", "--serve-arg=0.0.0.0")
    try:
        assert up["started"] is True and up["url"] == f"http://127.0.0.1:{port}"
        assert up["task"]  # generated ownership receipt
        argv = _cmdline(up["pid"])
        hosts = [argv[i + 1] for i, w in enumerate(argv) if w == "--host"]
        assert hosts[-1] == "127.0.0.1"
        assert worker_cli._is_harness(up["url"])
        for name in (f"launcher-{port}.json", f"harness-{port}.log", f"harness-{port}.db"):
            assert _mode(state / name) == 0o600, name

        # A second launcher using the defaults reuses it, gets its own task
        # id, and its down does not stop the first launcher's harness.
        again = _cli(capsys, "up", *base)
        assert again["started"] is False and again["task"] != up["task"]
        down_other = _cli(capsys, "down", *base, "--task", again["task"])
        assert down_other["stopped"] is False
        assert worker_cli._is_harness(up["url"])

        down = _cli(capsys, "down", *base, "--task", up["task"], "--timeout", "10")
        assert down["stopped"] is True and down["drained"]["pending"] == 0
        assert not worker_cli._is_harness(up["url"])
    finally:
        if worker_cli._pid_alive(up["pid"]):
            subprocess.run(["kill", str(up["pid"])], check=False)


def test_launcher_never_signals_a_recycled_pid(tmp_path, capsys) -> None:
    # A stale record whose pid now belongs to another process (different
    # start time) must not be signalled.
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        port = _free_port()
        (tmp_path / f"launcher-{port}.json").write_text(json.dumps(
            {"pid": sleeper.pid, "start_time": "1", "task": "t", "port": port}
        ))
        out = _cli(capsys, "down", "--port", str(port), "--state-dir", str(tmp_path), "--task", "t")
        assert out["stopped"] is False
        assert sleeper.poll() is None
    finally:
        sleeper.kill()
        sleeper.wait()


def test_launcher_refuses_to_reuse_a_harness_that_cannot_deliver(tmp_path, capsys) -> None:
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "agent_harness.cli", "serve", "--port", str(port), "--no-observer"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        url = f"http://127.0.0.1:{port}"
        for _ in range(200):
            try:
                worker_cli._request(f"{url}/v1/backends", timeout=0.5)
                break
            except worker_cli.HarnessError:
                time.sleep(0.05)
        from agent_harness.cli import main

        assert main(["up", "--port", str(port), "--state-dir", str(tmp_path)]) == 1
        assert "does not execute runs" in capsys.readouterr().err
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_launcher_down_without_up_stops_nothing(tmp_path, capsys) -> None:
    out = _cli(capsys, "down", "--port", str(_free_port()), "--state-dir", str(tmp_path), "--task", "x")
    assert out["stopped"] is False


def test_launcher_tightens_modes_of_its_own_default_state(tmp_path, capsys, monkeypatch) -> None:
    import os

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    port = _free_port()
    state = tmp_path / "agent-harness"
    state.mkdir(mode=0o755)
    os.chmod(state, 0o755)
    for name in (f"harness-{port}.db", f"harness-{port}.log"):
        (state / name).write_bytes(b"")
        os.chmod(state / name, 0o644)
    up = _cli(capsys, "up", "--port", str(port), "--serve-arg=--no-observer")
    try:
        assert _mode(state) == 0o700
        for name in (f"harness-{port}.db", f"harness-{port}.log", f"launcher-{port}.json"):
            assert _mode(state / name) == 0o600, name
    finally:
        _cli(capsys, "down", "--port", str(port), "--task", up["task"], "--timeout", "10")
