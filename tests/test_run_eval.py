"""Runner tests. Every request goes to an in-process httpx.MockTransport: no network, no API key."""

import asyncio
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

import run_eval

ROOT = Path(__file__).resolve().parents[1]
KEY = "sk-test-not-a-real-key"
ITEMS = [{"id": "x1", "task": "nli", "prompt": "p1"}, {"id": "x2", "task": "nli", "prompt": "p2"}]
Handler = Callable[[httpx.Request], httpx.Response]


def openai_ok(content: str | None = "neutral", finish: str = "stop") -> httpx.Response:
    body = {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1},
    }
    return httpx.Response(200, json=body)


def anthropic_ok(text: str = "neutral", stop: str = "end_turn") -> httpx.Response:
    content = [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": text}]
    return httpx.Response(200, json={"content": content, "stop_reason": stop, "usage": {"input_tokens": 5}})


class Endpoint:
    """Records requests and answers them from a script of responses (the last one repeats)."""

    def __init__(self, *script: httpx.Response | Exception) -> None:
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self.script[min(len(self.requests), len(self.script)) - 1]
        if isinstance(step, Exception):
            raise step
        return step


class Sleeps:
    def __init__(self) -> None:
        self.seconds: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.seconds.append(seconds)


def cfg(provider: str = "openai", *opts: str) -> run_eval.Config:
    return run_eval.Config(provider, "m", "http://endpoint.test/v1", KEY, frozenset(opts))


def one_item(endpoint: Handler, sleeps: Sleeps, config: run_eval.Config | None = None) -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.MockTransport(endpoint)) as client:
            return await run_eval.run_item(client, config or cfg(), ITEMS[0], sleeps)

    return asyncio.run(go())


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("BN_EVAL_BASE_URL", "http://endpoint.test/v1/")
    monkeypatch.setenv("BN_EVAL_API_KEY", KEY)
    (tmp_path / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in ITEMS), encoding="utf-8")
    return tmp_path


def cli(workdir: Path, endpoint: Handler, *args: str, sleeps: Sleeps | None = None) -> int:
    argv = args or ("openai", "m", str(workdir / "out.jsonl"))
    transport = httpx.MockTransport(endpoint)
    return run_eval.main(argv, items_path=workdir / "items.jsonl", transport=transport, sleep=sleeps or Sleeps())


def rows(workdir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (workdir / "out.jsonl").read_text(encoding="utf-8").splitlines()]


# --- request shape: the published v1.0.0 run settings -------------------------------------------


def test_request_bodies_match_v1_settings() -> None:
    url, headers, body = run_eval.build_request(cfg("anthropic"), "p")
    assert url == "http://endpoint.test/v1/messages"
    assert headers == {"x-api-key": KEY, "anthropic-version": "2023-06-01"}
    assert json.dumps(body) == (
        '{"model": "m", "max_tokens": 512, "messages": [{"role": "user", "content": "p"}], "temperature": 0}'
    )
    assert "temperature" not in run_eval.build_request(cfg("anthropic", "notemp"), "p")[2]

    url, headers, body = run_eval.build_request(cfg("openai"), "p")
    assert url == "http://endpoint.test/v1/chat/completions" and headers == {"Authorization": f"Bearer {KEY}"}
    assert json.dumps(body) == (
        '{"model": "m", "messages": [{"role": "user", "content": "p"}], "max_tokens": 512, "temperature": 0}'
    )
    assert run_eval.build_request(cfg("openai", "mct", "notemp"), "p")[2] == {
        "model": "m",
        "messages": [{"role": "user", "content": "p"}],
        "max_completion_tokens": 2048,
    }


def test_config_repr_hides_the_key() -> None:
    assert KEY not in repr(cfg())


# --- stop reason and empty answers (truncation used to be silent) --------------------------------


def test_stop_reason_recorded() -> None:
    row = one_item(Endpoint(anthropic_ok("cut off mid", stop="max_tokens")), Sleeps(), cfg("anthropic"))
    assert row["output"] == "cut off mid" and row["stop_reason"] == "max_tokens"
    assert set(row) == {"id", "output", "usage", "latency_s", "stop_reason"}
    assert one_item(Endpoint(openai_ok(finish="length")), Sleeps())["stop_reason"] == "length"


def test_null_content_is_flagged(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli(workdir, Endpoint(openai_ok(content=None, finish="length"))) == 0
    assert [(r["output"], r["stop_reason"]) for r in rows(workdir)] == [("", "length")] * 2
    assert "2 answer(s) stopped at the token limit, 2 empty" in capsys.readouterr().err


# --- retry policy ---------------------------------------------------------------------------------


def test_non_retryable_status_fails_fast() -> None:
    endpoint, sleeps = Endpoint(httpx.Response(401, json={"error": "bad key"})), Sleeps()
    assert one_item(endpoint, sleeps) == {"id": "x1", "error": "HTTPStatusError 401"}
    assert len(endpoint.requests) == 1 and sleeps.seconds == []


def test_retries_429_and_5xx() -> None:
    endpoint, sleeps = Endpoint(httpx.Response(429), httpx.Response(503), openai_ok()), Sleeps()
    row = one_item(endpoint, sleeps)
    assert row["output"] == "neutral" and len(endpoint.requests) == 3 and sleeps.seconds == [2.0, 4.0]


def test_retry_after_is_honoured_and_capped() -> None:
    hinted = Endpoint(httpx.Response(429, headers={"retry-after": "7"}), openai_ok())
    sleeps = Sleeps()
    one_item(hinted, sleeps)
    capped = Endpoint(httpx.Response(429, headers={"retry-after": "3600"}), openai_ok())
    one_item(capped, sleeps)
    garbage = Endpoint(httpx.Response(429, headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), openai_ok())
    one_item(garbage, sleeps)
    assert sleeps.seconds == [7.0, run_eval.MAX_RETRY_AFTER_S, 2.0]


def test_timeout_retried_then_recorded() -> None:
    endpoint, sleeps = Endpoint(httpx.ReadTimeout("slow")), Sleeps()
    assert one_item(endpoint, sleeps) == {"id": "x1", "error": "ReadTimeout"}
    assert len(endpoint.requests) == run_eval.MAX_ATTEMPTS
    recovered = Endpoint(httpx.ConnectError("down"), openai_ok())
    assert one_item(recovered, Sleeps())["output"] == "neutral"


def test_no_sleep_after_final_attempt() -> None:
    endpoint, sleeps = Endpoint(httpx.Response(500)), Sleeps()
    assert one_item(endpoint, sleeps) == {"id": "x1", "error": "HTTPStatusError 500"}
    assert len(endpoint.requests) == 3 and sleeps.seconds == [2.0, 4.0]


def test_malformed_response_is_recorded_not_retried() -> None:
    endpoint, sleeps = Endpoint(httpx.Response(200, json={"choices": []})), Sleeps()
    assert one_item(endpoint, sleeps) == {"id": "x1", "error": "IndexError"}
    assert len(endpoint.requests) == 1
    not_json = Endpoint(httpx.Response(200, text="<html>gateway</html>"), openai_ok())
    assert one_item(not_json, Sleeps())["output"] == "neutral"


def test_at_most_six_requests_in_flight(workdir: Path) -> None:
    items = [{"id": f"i{n}", "prompt": "p"} for n in range(40)]
    (workdir / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    state = {"now": 0, "peak": 0}

    async def slow(request: httpx.Request) -> httpx.Response:
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
        await asyncio.sleep(0.001)
        state["now"] -= 1
        return openai_ok()

    transport = httpx.MockTransport(slow)
    argv = ["openai", "m", str(workdir / "out.jsonl")]
    assert run_eval.main(argv, items_path=workdir / "items.jsonl", transport=transport) == 0
    assert len(rows(workdir)) == 40 and 1 < state["peak"] <= run_eval.CONCURRENCY


# --- resume ---------------------------------------------------------------------------------------


def test_resume_retries_errored_items(workdir: Path) -> None:
    """An item that ended in an error used to be skipped for ever on re-run."""
    assert cli(workdir, Endpoint(httpx.Response(401))) == 1
    assert [r.get("error") for r in rows(workdir)] == ["HTTPStatusError 401"] * 2
    healthy = Endpoint(openai_ok())
    assert cli(workdir, healthy) == 0
    assert len(healthy.requests) == 2
    assert run_eval.completed_ids(rows(workdir)) == {"x1", "x2"}
    again = Endpoint(openai_ok())
    assert cli(workdir, again) == 0 and again.requests == []


def test_resume_drops_half_written_last_line(workdir: Path) -> None:
    good = '{"id": "x1", "output": "neutral", "usage": {}, "latency_s": 0.1}\n'
    (workdir / "out.jsonl").write_text(good + '{"id": "x2", "outp', encoding="utf-8")
    endpoint = Endpoint(openai_ok())
    assert cli(workdir, endpoint) == 0
    assert len(endpoint.requests) == 1
    text = (workdir / "out.jsonl").read_text(encoding="utf-8")
    assert text.startswith(good) and [r["id"] for r in rows(workdir)] == ["x1", "x2"]


def test_resume_terminates_a_complete_last_line(workdir: Path) -> None:
    (workdir / "out.jsonl").write_text('{"id": "x1", "output": "neutral"}', encoding="utf-8")
    assert cli(workdir, Endpoint(openai_ok())) == 0
    assert [r["id"] for r in rows(workdir)] == ["x1", "x2"]


def test_blank_lines_ignored(workdir: Path) -> None:
    with (workdir / "items.jsonl").open("a", encoding="utf-8") as f:
        f.write("\n\n")
    assert cli(workdir, Endpoint(openai_ok())) == 0
    assert len(rows(workdir)) == 2


# --- command line ---------------------------------------------------------------------------------


def test_exit_code_nonzero_when_items_failed(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli(workdir, Endpoint(httpx.Response(401))) == 1
    out = capsys.readouterr()
    assert "0 answered, 2 errors" in out.out and "2 item(s) ended in an error" in out.err
    assert cli(workdir, Endpoint(openai_ok())) == 0
    assert "2 answered, 0 errors" in capsys.readouterr().out


def test_error_rows_never_contain_the_key(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli(workdir, Endpoint(httpx.Response(403), httpx.ConnectError(f"refused {KEY}")))
    captured = capsys.readouterr()
    assert KEY not in (workdir / "out.jsonl").read_text(encoding="utf-8") + captured.out + captured.err


def test_cli_rejects_unknown_provider_and_option(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    endpoint = Endpoint(openai_ok())
    out = str(workdir / "out.jsonl")
    for argv in (["anthropc", "m", out], ["openai", "m", out, "no-temp"], ["openai", "m"], ["openai", "", out]):
        with pytest.raises(SystemExit) as stop:
            cli(workdir, endpoint, *argv)
        assert stop.value.code == 2
    assert "unknown provider 'anthropc'" in capsys.readouterr().err
    assert endpoint.requests == [] and not (workdir / "out.jsonl").exists()


def test_missing_env_is_clear_error(
    workdir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("BN_EVAL_API_KEY")
    with pytest.raises(SystemExit) as stop:
        cli(workdir, Endpoint(openai_ok()))
    assert stop.value.code == 2 and "BN_EVAL_BASE_URL and BN_EVAL_API_KEY" in capsys.readouterr().err


def test_base_url_trailing_slash_and_anthropic_path(workdir: Path) -> None:
    endpoint = Endpoint(anthropic_ok())
    assert cli(workdir, endpoint, "anthropic", "m", str(workdir / "out.jsonl")) == 0
    assert {str(r.url) for r in endpoint.requests} == {"http://endpoint.test/v1/messages"}
    assert [r["output"] for r in rows(workdir)] == ["neutral", "neutral"]


def test_import_has_no_side_effects(tmp_path: Path) -> None:
    """The old script read sys.argv and the environment at import time."""
    code = "import sys; sys.path.insert(0, sys.argv[1]); import run_eval"
    done = subprocess.run([sys.executable, "-c", code, str(ROOT)], cwd=tmp_path, env={}, capture_output=True)
    assert done.returncode == 0, done.stderr
    assert list(tmp_path.iterdir()) == []
