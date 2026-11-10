"""Run every item against one model through an OpenAI- or Anthropic-compatible endpoint.

Usage: run_eval.py <provider> <model> <out.jsonl> [notemp] [mct]

  provider  anthropic (POST <base>/messages) or openai (POST <base>/chat/completions)
  notemp    do not send temperature 0 (for endpoints that reject it)
  mct       OpenAI only: send max_completion_tokens 2048 instead of max_tokens 512

The endpoint comes from BN_EVAL_BASE_URL and BN_EVAL_API_KEY. One JSON line is appended to the
output file per item: the answer with usage, latency and the provider's stop reason, or an error
name. Re-running with the same output file resumes: items whose latest row is an answer are
skipped, items whose latest row is an error are tried again. The exit status is 1 if any item
ended in an error.
"""

import asyncio
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, NoReturn

import httpx

ROOT = Path(__file__).resolve().parent
ITEMS_PATH = ROOT / "items.jsonl"
USAGE = "usage: run_eval.py <provider> <model> <out.jsonl> [notemp] [mct]"
PROVIDERS = ("anthropic", "openai")
OPTIONS = frozenset({"notemp", "mct"})
CONCURRENCY = 6
MAX_ATTEMPTS = 3
TIMEOUT_S = 120.0
MAX_RETRY_AFTER_S = 30.0
TRUNCATED = frozenset({"max_tokens", "length"})  # Anthropic stop_reason, OpenAI finish_reason
# A 200 response that does not have the expected shape. Deterministic, so not retried.
BAD_RESPONSE = (KeyError, IndexError, TypeError, AttributeError)

Record = dict[str, Any]
Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class Config:
    provider: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    opts: frozenset[str] = frozenset()


def fail(message: str) -> NoReturn:
    print(f"run_eval.py: {message}", file=sys.stderr)
    raise SystemExit(2)


def load_jsonl(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def repair_tail(path: Path) -> str | None:
    """Make sure the run file ends on a line break before rows are appended to it.

    A process killed mid-write leaves a half-written last line; it is dropped (that item runs again).
    Returns a description of what was done, or None if the file was already clean.
    """
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return None
    head, _, tail = data.rpartition(b"\n")
    try:
        json.loads(tail)
    except ValueError:
        fixed, action = (head + b"\n" if head else b""), "dropped a half-written last line"
    else:
        fixed, action = data + b"\n", "terminated the last line"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(fixed)
    tmp.replace(path)
    return action


def completed_ids(rows: Iterable[Record]) -> set[str]:
    """Ids whose latest row is an answer. An item whose latest row is an error is run again."""
    latest: dict[str, bool] = {}
    for r in rows:
        latest[r["id"]] = not r.get("error")
    return {rid for rid, answered in latest.items() if answered}


def build_request(cfg: Config, prompt: str) -> tuple[str, dict[str, str], Record]:
    """URL, headers and JSON body for one item. These are the published v1.0.0 run settings."""
    messages = [{"role": "user", "content": prompt}]
    body: Record
    if cfg.provider == "anthropic":
        body = {"model": cfg.model, "max_tokens": 512, "messages": messages}
        if "notemp" not in cfg.opts:
            body["temperature"] = 0
        return cfg.base_url + "/messages", {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"}, body
    body = {"model": cfg.model, "messages": messages}
    if "mct" in cfg.opts:
        body["max_completion_tokens"] = 2048
    else:
        body["max_tokens"] = 512
    if "notemp" not in cfg.opts:
        body["temperature"] = 0
    return cfg.base_url + "/chat/completions", {"Authorization": f"Bearer {cfg.api_key}"}, body


def parse_response(provider: str, data: Record) -> tuple[str, Record, str | None]:
    """Answer text, usage and the provider's stop reason."""
    if provider == "anthropic":
        text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
        return text, data.get("usage", {}), data.get("stop_reason")
    choice = data["choices"][0]
    return choice["message"]["content"] or "", data.get("usage", {}), choice.get("finish_reason")


def describe(exc: Exception) -> str:
    """What goes in an error row: the exception name and HTTP status, never headers, URL or key."""
    status = f" {exc.response.status_code}" if isinstance(exc, httpx.HTTPStatusError) else ""
    return type(exc).__name__ + status


def is_retryable(exc: Exception) -> bool:
    """Timeouts, connection failures, garbled bodies, and 408/409/429/5xx are worth another attempt."""
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status in (408, 409, 429) or status >= 500
    if isinstance(exc, httpx.UnsupportedProtocol):
        return False
    return isinstance(exc, httpx.TransportError | json.JSONDecodeError)


def backoff_s(exc: Exception, attempt: int) -> float:
    """Seconds to wait after a failed attempt: 2 s, then 4 s, or the server's Retry-After (capped)."""
    base = 2.0 * attempt
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            hinted = float(exc.response.headers.get("retry-after", ""))
        except ValueError:
            return base
        return min(max(base, hinted), MAX_RETRY_AFTER_S)
    return base


async def call(client: httpx.AsyncClient, cfg: Config, item: Record) -> tuple[str, Record, str | None]:
    url, headers, body = build_request(cfg, item["prompt"])
    r = await client.post(url, json=body, headers=headers)
    r.raise_for_status()
    return parse_response(cfg.provider, r.json())


async def run_item(client: httpx.AsyncClient, cfg: Config, item: Record, sleep: Sleep = asyncio.sleep) -> Record:
    """One item, up to MAX_ATTEMPTS attempts. Returns the row to append: an answer or an error name."""
    error = "NotAttempted"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.perf_counter()
        try:
            text, usage, stop = await call(client, cfg, item)
        except (httpx.HTTPError, httpx.InvalidURL, json.JSONDecodeError, *BAD_RESPONSE) as exc:
            error = describe(exc)
            if attempt == MAX_ATTEMPTS or not is_retryable(exc):
                break
            await sleep(backoff_s(exc, attempt))
        else:
            latency = round(time.perf_counter() - start, 2)
            return {"id": item["id"], "output": text, "usage": usage, "latency_s": latency, "stop_reason": stop}
    return {"id": item["id"], "error": error}


async def run(
    cfg: Config,
    items: Sequence[Record],
    sink: IO[str],
    client: httpx.AsyncClient,
    sleep: Sleep = asyncio.sleep,
) -> dict[str, int]:
    """Run the items, CONCURRENCY in flight, appending one row each to sink as it finishes."""
    sem = asyncio.Semaphore(CONCURRENCY)
    counts = {"answered": 0, "errors": 0, "truncated": 0, "empty": 0}

    async def one(item: Record) -> None:
        async with sem:
            row = await run_item(client, cfg, item, sleep)
        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
        sink.flush()
        if row.get("error"):
            counts["errors"] += 1
            return
        counts["answered"] += 1
        counts["truncated"] += row["stop_reason"] in TRUNCATED
        counts["empty"] += not row["output"].strip()

    await asyncio.gather(*(one(i) for i in items))
    return counts


def parse_args(argv: Sequence[str]) -> tuple[str, str, Path, frozenset[str]]:
    if len(argv) < 3:
        fail(USAGE)
    provider, model, out, opts = argv[0], argv[1], Path(argv[2]), frozenset(argv[3:])
    if provider not in PROVIDERS:
        fail(f"unknown provider {provider!r}; expected one of: {', '.join(PROVIDERS)}\n{USAGE}")
    if not model:
        fail(f"model must not be empty\n{USAGE}")
    if opts - OPTIONS:
        fail(f"unknown option(s) {', '.join(sorted(opts - OPTIONS))}; known: {', '.join(sorted(OPTIONS))}\n{USAGE}")
    return provider, model, out, opts


def main(
    argv: Sequence[str] | None = None,
    *,
    items_path: Path = ITEMS_PATH,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Sleep = asyncio.sleep,
) -> int:
    provider, model, out, opts = parse_args(sys.argv[1:] if argv is None else argv)
    base, key = os.environ.get("BN_EVAL_BASE_URL", ""), os.environ.get("BN_EVAL_API_KEY", "")
    if not base or not key:
        fail("set BN_EVAL_BASE_URL and BN_EVAL_API_KEY for the endpoint to call")
    cfg = Config(provider, model, base.rstrip("/"), key, opts)
    items = load_jsonl(items_path)
    done: set[str] = set()
    if out.exists():
        repaired = repair_tail(out)
        if repaired:
            print(f"run_eval.py: {out}: {repaired}", file=sys.stderr)
        done = completed_ids(load_jsonl(out))
    todo = [i for i in items if i["id"] not in done]

    async def go(sink: IO[str]) -> dict[str, int]:
        async with httpx.AsyncClient(timeout=TIMEOUT_S, transport=transport) as client:
            return await run(cfg, todo, sink, client, sleep)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8", newline="\n") as sink:
        counts = asyncio.run(go(sink))
    print(f"DONE {out}: {counts['answered']} answered, {counts['errors']} errors, {len(done)} already done")
    if counts["truncated"] or counts["empty"]:
        print(
            f"run_eval.py: warning: {counts['truncated']} answer(s) stopped at the token limit, "
            f"{counts['empty']} empty; see stop_reason in {out}",
            file=sys.stderr,
        )
    if counts["errors"]:
        print(f"run_eval.py: {counts['errors']} item(s) ended in an error; re-run to try them again", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
