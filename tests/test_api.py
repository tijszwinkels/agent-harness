import asyncio

import pytest
from fastapi.testclient import TestClient

from agent_harness.api import _materialize_run_result, _replay_after, _sse_stream, create_app
from agent_harness.events import DurableEventBus, InMemoryEventBus
from agent_harness.models import CreateRunRequest, CreateSessionRequest, Event, Project, Session
from agent_harness.orchestrator import ProcessCommand, RunManager, RunProcessResult, SubmitResult
from agent_harness.repository import InMemoryRepository
from agent_harness.settings import ObserverSettings
from agent_harness.storage import open_sqlite_repository


def test_health_and_backend_listing() -> None:
    client = TestClient(create_app())

    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/v1/health").json() == {"status": "ok"}

    response = client.get("/v1/backends")
    assert response.status_code == 200
    assert {item["name"] for item in response.json()["data"]} == {"claude-code", "codex"}

    assert client.get("/v1/backends/codex/models").json() == {"data": []}
    assert client.get("/v1/backends/claude-code/models").json() == {"data": []}
    assert client.get("/v1/backends/unknown/models").status_code == 404


def test_session_create_list_get_archive_flow() -> None:
    client = TestClient(create_app())
    payload = {
        "backend": "codex",
        "model": "gpt-5.4",
        "project": {"path": "/tmp/proj", "name": "proj"},
        "title": "Implement scaffold",
    }

    create_response = client.post("/v1/sessions", json=payload)
    assert create_response.status_code == 201
    session = create_response.json()
    assert session["backend"] == "codex"
    assert session["status"] == "idle"
    assert session["origin"] == "harness"

    list_response = client.get("/v1/sessions")
    assert list_response.status_code == 200
    assert [item["id"] for item in list_response.json()["data"]] == [session["id"]]

    get_response = client.get(f"/v1/sessions/{session['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == session["id"]

    archive_response = client.delete(f"/v1/sessions/{session['id']}")
    assert archive_response.status_code == 200
    assert archive_response.json()["status"] == "archived"


def test_session_create_rejects_unknown_backend() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/sessions",
        json={"backend": "unknown", "model": "x", "project": {"path": "/tmp", "name": "tmp"}},
    )

    assert response.status_code == 422


def test_run_create_list_get_interrupt_flow() -> None:
    client = TestClient(create_app())
    session = client.post(
        "/v1/sessions",
        json={
            "backend": "codex",
            "model": "gpt-5.4",
            "project": {"path": "/tmp/proj", "name": "proj"},
        },
    ).json()

    create_response = client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "hello"})
    assert create_response.status_code == 202
    created = create_response.json()
    assert created["session_id"] == session["id"]
    assert created["run_id"].startswith("run_")

    list_response = client.get(f"/v1/sessions/{session['id']}/runs")
    assert list_response.status_code == 200
    run = list_response.json()["data"][0]
    assert run["id"] == created["run_id"]
    assert run["status"] == "running"
    assert run["origin"] == "harness"

    get_response = client.get(f"/v1/sessions/{session['id']}/runs/{created['run_id']}")
    assert get_response.status_code == 200
    assert get_response.json()["id"] == created["run_id"]

    interrupt_response = client.delete(f"/v1/sessions/{session['id']}/runs/{created['run_id']}")
    assert interrupt_response.status_code == 200
    body = interrupt_response.json()
    assert body["run"]["status"] == "interrupted"
    assert body["run"]["stop_reason"] == "interrupted"
    assert body["dropped_queued"] == []


def test_run_create_starts_run_manager_when_configured() -> None:
    class FakeRunManager:
        def __init__(self) -> None:
            self.submitted = []

        def submit(self, *, session, run, command, on_start=None):
            self.submitted.append((session, run, command, on_start))
            # Mirror real RunManager behavior: invoke on_start when accepted
            # as "running" so api.py flips the repo into the matching state.
            if on_start is not None:
                on_start()
            return SubmitResult(accepted=True, status="running")

    class FakeBuilder:
        def build(self, *, session, run, message, is_first_run=True):
            return ("fake", session.id, run.id, message.id, is_first_run)

    manager = FakeRunManager()
    client = TestClient(
        create_app(
            run_manager=manager,
            command_builders={"codex": FakeBuilder()},
        )
    )
    session = client.post(
        "/v1/sessions",
        json={
            "backend": "codex",
            "model": "gpt-5.4",
            "project": {"path": "/tmp/proj", "name": "proj"},
        },
    ).json()

    response = client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "hello"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "running"
    assert len(manager.submitted) == 1
    started_session, started_run, command, _on_start = manager.submitted[0]
    assert started_session.id == session["id"]
    assert started_run.id == body["run_id"]
    assert command[0] == "fake"


@pytest.mark.asyncio
async def test_run_create_schedules_materializer_even_if_repo_start_run_fails() -> None:
    """Regression: ``on_start`` used to call ``repo.start_run`` and
    ``_schedule_run_result_materialization`` back-to-back. If the first
    raised, the orchestrator's swallow-and-log caught it and the
    materializer was never scheduled — leaving the run stuck at
    ``queued`` in the repo even after the process finished.

    The fix isolates the two calls so materializer scheduling always
    happens, regardless of whether the start-status persistence raised.
    """
    captured: dict = {}

    class FakeRunManager:
        async def wait(self, run_id):
            return RunProcessResult(run_id=run_id, status="completed", returncode=0)

        def submit(self, *, session, run, command, on_start=None):
            # Mirror real RunManager: invoke on_start to trigger the
            # repo.start_run + materializer scheduling path under test.
            if on_start is not None:
                on_start()
            return SubmitResult(accepted=True, status="running")

    class RaisingRepo(InMemoryRepository):
        def start_run(self, session_id, run_id):
            captured["start_run_called"] = True
            raise RuntimeError("simulated repo failure")

    class FakeBuilder:
        def build(self, *, session, run, message, is_first_run=True):
            return ("fake", session.id, run.id, message.id, is_first_run)

    repo = RaisingRepo()
    client = TestClient(
        create_app(
            repository=repo,
            run_manager=FakeRunManager(),
            command_builders={"codex": FakeBuilder()},
        )
    )
    session = client.post(
        "/v1/sessions",
        json={"backend": "codex", "model": "gpt-5.4", "project": {"path": "/tmp/proj", "name": "proj"}},
    ).json()
    response = client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "hello"})

    assert response.status_code == 202
    assert captured.get("start_run_called") is True

    # Materializer is fire-and-forget — give it the event loop tick to
    # call finish_run, then confirm the run reached "completed". If the
    # bug were present, the run would stay "queued".
    run_id = response.json()["run_id"]
    for _ in range(10):
        stored = repo.get_run(session["id"], run_id)
        if stored.status == "completed":
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError(f"run stuck at status={stored.status} — materializer did not run")


def test_run_create_returns_409_when_external_session_cannot_be_resumed() -> None:
    class FakeRunManager:
        def submit(self, *, session, run, command, on_start=None):  # pragma: no cover - should not be reached
            raise AssertionError("run manager should not submit when command build fails")

        def drop_queued(self, session_id):  # pragma: no cover - not exercised
            return []

        async def interrupt(self, session_id, run_id):  # pragma: no cover - not exercised
            return False

    repo = InMemoryRepository()
    repo.upsert_session(
        Session(
            id="external_without_backend_prefix",
            backend="codex",
            model="gpt-5.4",
            project=Project(path="/tmp/proj", name="proj"),
            origin="external",
        )
    )
    client = TestClient(create_app(repository=repo, run_manager=FakeRunManager()))

    response = client.post("/v1/sessions/external_without_backend_prefix/runs", json={"message": "hello"})

    assert response.status_code == 409
    assert response.json()["detail"] == "Cannot resume external codex session from id external_without_backend_prefix"
    assert repo.list_runs("external_without_backend_prefix") == []
    assert repo.list_messages("external_without_backend_prefix") == []


@pytest.mark.asyncio
async def test_run_result_materialization_updates_repository_status() -> None:
    repo = InMemoryRepository()
    session = repo.create_session(
        CreateSessionRequest(
            backend="codex",
            model="gpt-5.4-mini",
            project=Project(path="/tmp/proj", name="proj"),
        )
    )
    run = repo.create_run(session.id, CreateRunRequest(message="hello"))

    async def wait(run_id):
        return RunProcessResult(run_id=run_id, status="completed", returncode=0)

    await _materialize_run_result(wait, repo.finish_run, session_id=session.id, run_id=run.id)

    stored = repo.get_run(session.id, run.id)
    assert stored.status == "completed"
    assert stored.stop_reason == "end_turn"
    assert stored.completed_at is not None
    assert repo.get_session(session.id).status == "idle"


def test_run_create_surfaces_queued_status_when_run_manager_queues() -> None:
    # When the RunManager reports the run is queued (because another run is
    # in-flight for the same session) the 202 body must surface that so the
    # bridge can log / coalesce / show the user "your message is queued".
    class FakeRunManager:
        def submit(self, *, session, run, command, on_start=None):
            return SubmitResult(accepted=True, status="queued")

    class FakeBuilder:
        def build(self, *, session, run, message, is_first_run=True):
            return ("fake",)

    client = TestClient(
        create_app(
            run_manager=FakeRunManager(),
            command_builders={"codex": FakeBuilder()},
        )
    )
    session = client.post(
        "/v1/sessions",
        json={"backend": "codex", "model": "gpt-5.4", "project": {"path": "/tmp/p", "name": "p"}},
    ).json()

    response = client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "hi"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    # And the repo reflects the still-queued state — no on_start was called.
    list_response = client.get(f"/v1/sessions/{session['id']}/runs")
    assert list_response.json()["data"][0]["status"] == "queued"


def test_run_create_returns_429_when_run_manager_rejects_queue_full() -> None:
    class FakeRunManager:
        def submit(self, *, session, run, command, on_start=None):
            return SubmitResult(accepted=False, status=None, reason="queue_full")

    class FakeBuilder:
        def build(self, *, session, run, message, is_first_run=True):
            return ("fake",)

    client = TestClient(
        create_app(
            run_manager=FakeRunManager(),
            command_builders={"codex": FakeBuilder()},
        )
    )
    session = client.post(
        "/v1/sessions",
        json={"backend": "codex", "model": "gpt-5.4", "project": {"path": "/tmp/p", "name": "p"}},
    ).json()

    response = client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "hi"})

    assert response.status_code == 429
    assert "queue is full" in response.json()["detail"].lower()
    # The rejected run is recorded as ``failed`` so the session doesn't carry
    # a phantom "queued" run forever.
    list_response = client.get(f"/v1/sessions/{session['id']}/runs")
    assert list_response.json()["data"][0]["status"] == "failed"


def test_interrupt_run_drops_queued_runs_and_surfaces_them_in_response() -> None:
    # Real flow: two rapid POST /runs land — first is "running", second is
    # "queued". DELETE on the running run must:
    #   1) terminate the active subprocess
    #   2) drop the queued run from the orchestrator + flip it to interrupted
    #      in the repo
    #   3) return both in the response so the bridge can tell the user which
    #      follow-up messages got cancelled.
    submitted: list[tuple[str, str]] = []

    class FakeRunManager:
        def __init__(self) -> None:
            self._active_session: str | None = None
            self._queue_run_ids: list[str] = []
            self.interrupt_calls: list[tuple[str, str]] = []

        def submit(self, *, session, run, command, on_start=None):
            submitted.append((session.id, run.id))
            if self._active_session is None:
                self._active_session = session.id
                if on_start is not None:
                    on_start()
                return SubmitResult(accepted=True, status="running")
            self._queue_run_ids.append(run.id)
            return SubmitResult(accepted=True, status="queued")

        def drop_queued(self, session_id):
            popped = list(self._queue_run_ids)
            self._queue_run_ids.clear()
            return popped

        async def interrupt(self, session_id, run_id):
            self.interrupt_calls.append((session_id, run_id))
            return True

    class FakeBuilder:
        def build(self, *, session, run, message, is_first_run=True):
            return ("fake",)

    manager = FakeRunManager()
    client = TestClient(
        create_app(
            run_manager=manager,
            command_builders={"codex": FakeBuilder()},
        )
    )
    session = client.post(
        "/v1/sessions",
        json={"backend": "codex", "model": "gpt-5.4", "project": {"path": "/tmp/p", "name": "p"}},
    ).json()

    first = client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "a"}).json()
    second = client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "b"}).json()
    third = client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "c"}).json()
    assert first["status"] == "running"
    assert second["status"] == "queued"
    assert third["status"] == "queued"

    response = client.delete(f"/v1/sessions/{session['id']}/runs/{first['run_id']}")

    assert response.status_code == 200
    body = response.json()
    assert body["run"]["id"] == first["run_id"]
    assert body["run"]["status"] == "interrupted"
    assert body["run"]["stop_reason"] == "interrupted"
    dropped_ids = {r["id"] for r in body["dropped_queued"]}
    assert dropped_ids == {second["run_id"], third["run_id"]}
    for r in body["dropped_queued"]:
        assert r["status"] == "interrupted"
        assert r["stop_reason"] == "interrupted"
    # Manager observed both the queue drop and the subprocess interrupt.
    assert manager.interrupt_calls == [(session["id"], first["run_id"])]


def test_interrupt_queued_run_directly_drops_entire_queue() -> None:
    # A targeted DELETE on a *queued* run still empties the rest of the
    # queue (matches the agreed UX: any interrupt signals "stop everything").
    class FakeRunManager:
        def __init__(self) -> None:
            self._active: str | None = None
            self._queued: list[str] = []

        def submit(self, *, session, run, command, on_start=None):
            if self._active is None:
                self._active = session.id
                if on_start is not None:
                    on_start()
                return SubmitResult(accepted=True, status="running")
            self._queued.append(run.id)
            return SubmitResult(accepted=True, status="queued")

        def drop_queued(self, session_id):
            popped = list(self._queued)
            self._queued.clear()
            return popped

        async def interrupt(self, session_id, run_id):
            # Targeted run is queued, not running — interrupt returns False
            # and that's fine; the repo flip happens via drop_queued_runs.
            return False

    class FakeBuilder:
        def build(self, *, session, run, message, is_first_run=True):
            return ("fake",)

    client = TestClient(
        create_app(
            run_manager=FakeRunManager(),
            command_builders={"codex": FakeBuilder()},
        )
    )
    session = client.post(
        "/v1/sessions",
        json={"backend": "codex", "model": "gpt-5.4", "project": {"path": "/tmp/p", "name": "p"}},
    ).json()

    running = client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "a"}).json()
    queued1 = client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "b"}).json()
    queued2 = client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "c"}).json()

    # DELETE the *second-in-line* queued run; both queued runs must be
    # surfaced as interrupted (the target via ``run``, the sibling via
    # ``dropped_queued``).
    response = client.delete(f"/v1/sessions/{session['id']}/runs/{queued2['run_id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["run"]["id"] == queued2["run_id"]
    assert body["run"]["status"] == "interrupted"
    dropped_ids = {r["id"] for r in body["dropped_queued"]}
    assert dropped_ids == {queued1["run_id"]}

    # Verify the running run was NOT touched — DELETE was targeted at a
    # queued run, so the active subprocess keeps running.
    runs = {r["id"]: r for r in client.get(f"/v1/sessions/{session['id']}/runs").json()["data"]}
    assert runs[running["run_id"]]["status"] == "running"


@pytest.mark.asyncio
async def test_rapid_back_to_back_creates_serialize_through_real_run_manager() -> None:
    # Highest-fidelity test for the bug: wire a real RunManager with a fake
    # process factory through the real FastAPI app. Two rapid POST /runs on
    # the same session must result in exactly ONE subprocess being spawned
    # until the first run completes.
    #
    # We drive the app via ``httpx.AsyncClient`` + ``ASGITransport`` instead
    # of the sync ``TestClient`` because TestClient spins up a fresh event
    # loop per request — the orchestrator's ``_run_and_forget`` task gets
    # cancelled between calls, which clears ``_active_run_by_session`` and
    # masks the very serialization behavior under test.
    import httpx

    class _FakeStream:
        def __init__(self) -> None:
            self._q: asyncio.Queue[bytes] = asyncio.Queue()
            self._q.put_nowait(b"")

        async def readline(self) -> bytes:
            return await self._q.get()

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = _FakeStream()
            self.stderr = _FakeStream()
            self.returncode: int | None = None
            self._done = asyncio.Event()

        async def wait(self) -> int:
            await self._done.wait()
            return self.returncode if self.returncode is not None else 0

        def finish(self) -> None:
            self.returncode = 0
            self._done.set()

        def terminate(self) -> None:
            self.returncode = -15
            self._done.set()

    spawned: list[_FakeProcess] = []

    async def factory(command: ProcessCommand) -> _FakeProcess:
        proc = _FakeProcess()
        spawned.append(proc)
        return proc

    class _FakeBuilder:
        def build(self, *, session, run, message, is_first_run=True):
            return ProcessCommand(argv=("fake", "noop"))

    repo = InMemoryRepository()
    bus = InMemoryEventBus()
    manager = RunManager(event_bus=bus, process_factory=factory)
    app = create_app(
        repository=repo,
        event_bus=bus,
        run_manager=manager,
        command_builders={"codex": _FakeBuilder()},
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        session_resp = await client.post(
            "/v1/sessions",
            json={"backend": "codex", "model": "gpt-5.4", "project": {"path": "/tmp/p", "name": "p"}},
        )
        session = session_resp.json()

        first_resp = await client.post(
            f"/v1/sessions/{session['id']}/runs", json={"message": "first"},
        )
        second_resp = await client.post(
            f"/v1/sessions/{session['id']}/runs", json={"message": "second"},
        )
        first = first_resp.json()
        second = second_resp.json()

        assert first["status"] == "running"
        assert second["status"] == "queued"

        # The first subprocess must already have spawned; the second must not.
        await asyncio.sleep(0)
        assert len(spawned) == 1

        runs_list = (await client.get(f"/v1/sessions/{session['id']}/runs")).json()["data"]
        runs = {r["id"]: r for r in runs_list}
        assert runs[first["run_id"]]["status"] == "running"
        assert runs[second["run_id"]]["status"] == "queued"

        # Drain the first; the second must spawn and flip to running.
        spawned[0].finish()
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(spawned) == 2

        runs_list = (await client.get(f"/v1/sessions/{session['id']}/runs")).json()["data"]
        runs = {r["id"]: r for r in runs_list}
        assert runs[first["run_id"]]["status"] == "completed"
        assert runs[second["run_id"]]["status"] == "running"

        spawned[1].finish()
        for _ in range(10):
            await asyncio.sleep(0)


def test_session_messages_endpoint_returns_materialized_messages() -> None:
    client = TestClient(create_app())
    session = client.post(
        "/v1/sessions",
        json={
            "backend": "codex",
            "model": "gpt-5.4",
            "project": {"path": "/tmp/proj", "name": "proj"},
        },
    ).json()
    client.post(f"/v1/sessions/{session['id']}/runs", json={"message": "hello"})

    response = client.get(f"/v1/sessions/{session['id']}/messages")

    assert response.status_code == 200
    messages = response.json()["data"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["blocks"][0] == {"type": "text", "text": "hello"}


@pytest.mark.asyncio
async def test_sse_stream_emits_keepalive_comment_when_bus_idle() -> None:
    """Without periodic frames clients can't detect a silently dead stream
    or a stale-cursor situation. The SSE layer must inject ``:ka`` comment
    frames after each ``keepalive_seconds`` window of bus silence."""
    from agent_harness.models import Event

    bus = InMemoryEventBus()
    stream = _sse_stream(bus, keepalive_seconds=0.01).__aiter__()
    try:
        # Without keepalives the iterator would block forever; the timeout
        # is the assertion that the keepalive path fires.
        frame = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        assert frame.startswith(":")
        assert frame.endswith("\n\n")

        # Real events still get framed normally; the keepalive must not
        # cancel the subscription.
        published = await bus.publish(
            Event(event="run.started", session_id="ses_a", run_id="run_a", data={}),
        )
        event_frame = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        assert "event: run.started" in event_frame
        assert f"id: {published.sequence}" in event_frame
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_durable_sse_stream_replays_reopened_repository_from_beginning(tmp_path) -> None:
    db_path = tmp_path / "harness.db"
    repository = open_sqlite_repository(db_path)
    first_bus = DurableEventBus(repository)
    published = await first_bus.publish(
        Event(event="session.updated", session_id="ses_a", data={}),
    )
    repository.close()

    reopened = open_sqlite_repository(db_path)
    stream = _sse_stream(DurableEventBus(reopened), after=0, keepalive_seconds=1).__aiter__()
    try:
        frame = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        assert f"id: {published.sequence}" in frame
        assert "event: session.updated" in frame
    finally:
        await stream.aclose()
        reopened.close()


@pytest.mark.asyncio
async def test_replay_after_defaults_to_current_sequence_for_now(tmp_path) -> None:
    repository = open_sqlite_repository(tmp_path / "harness.db")
    bus = DurableEventBus(repository)
    await bus.publish(Event(event="session.updated", session_id="ses_a", data={}))

    try:
        assert await _replay_after(bus, after=0, from_="now") == 1
        assert await _replay_after(bus, after=0, from_="beginning") == 0
        assert await _replay_after(bus, after=7, from_="beginning") == 7
    finally:
        repository.close()


@pytest.mark.asyncio
async def test_replay_after_uses_max_sequence_for_now() -> None:
    class Bus:
        def __init__(self) -> None:
            self.session_id: str | None = None

        async def max_sequence(self, *, session_id: str | None = None) -> int:
            self.session_id = session_id
            return 42

        async def replay(self, *args, **kwargs):
            raise AssertionError("from=now must not replay stored events")

    bus = Bus()

    assert await _replay_after(bus, after=0, from_="now", session_id="ses_a") == 42
    assert bus.session_id == "ses_a"


def test_events_max_sequence_endpoint_reports_current_max(tmp_path) -> None:
    """Cheap synchronous probe so clients (e.g. mm-bridge) can detect a
    harness restart without opening an SSE stream and waiting for an idle
    window. Returns 0 on a fresh bus; reflects the global max after publishes;
    accepts ``session_id`` for a per-session max."""
    repository = open_sqlite_repository(tmp_path / "harness.db")
    bus = DurableEventBus(repository)
    try:
        client = TestClient(create_app(repository=repository, event_bus=bus))

        empty = client.get("/v1/events/max-sequence")
        assert empty.status_code == 200
        assert empty.json() == {"sequence": 0}

        asyncio.run(bus.publish(Event(event="session.updated", session_id="ses_a", data={})))
        asyncio.run(bus.publish(Event(event="message", session_id="ses_a", data={})))
        asyncio.run(bus.publish(Event(event="message", session_id="ses_b", data={})))

        all_resp = client.get("/v1/events/max-sequence")
        assert all_resp.status_code == 200
        assert all_resp.json() == {"sequence": 3}

        per_session = client.get("/v1/events/max-sequence", params={"session_id": "ses_a"})
        assert per_session.status_code == 200
        assert per_session.json() == {"sequence": 2}

        missing = client.get("/v1/events/max-sequence", params={"session_id": "ses_missing"})
        assert missing.status_code == 200
        assert missing.json() == {"sequence": 0}
    finally:
        repository.close()


def test_missing_session_returns_404() -> None:
    client = TestClient(create_app())

    response = client.get("/v1/sessions/ses_missing")

    assert response.status_code == 404


def test_observer_service_starts_and_stops_with_lifespan(tmp_path) -> None:
    root = tmp_path / "transcripts"
    root.mkdir()
    service = FakeWatchService()
    created_tasks = []

    def watch_service_factory(**kwargs):
        service.roots = kwargs["roots"]
        return service

    def task_factory(coro):
        import asyncio

        task = asyncio.create_task(coro)
        created_tasks.append(task)
        return task

    app = create_app(
        observer_settings=ObserverSettings.from_roots([root]),
        watch_service_factory=watch_service_factory,
        task_factory=task_factory,
    )

    with TestClient(app):
        assert service.started
        assert service.stop_event is not None
        assert created_tasks and not created_tasks[0].done()

    assert service.stopped
    assert created_tasks[0].done()
    assert service.roots == (root,)


def test_observer_startup_configuration_errors_are_logged_and_raised(tmp_path, caplog) -> None:
    missing = tmp_path / "missing"
    app = create_app(observer_settings=ObserverSettings.from_roots([missing]))

    with caplog.at_level("ERROR"):
        try:
            with TestClient(app):
                pass
        except Exception as exc:
            raised = exc
        else:
            raised = None

    assert raised is not None
    assert "Failed to start transcript observer service" in caplog.text
    assert "Observer root does not exist" in str(raised)


class FakeWatchService:
    def __init__(self) -> None:
        self.roots = ()
        self.started = False
        self.stopped = False
        self.stop_event = None

    async def watch_forever(self, *, stop_event=None) -> None:
        import asyncio

        self.started = True
        self.stop_event = stop_event
        while stop_event is not None and not stop_event.is_set():
            await asyncio.sleep(0)
        self.stopped = True
