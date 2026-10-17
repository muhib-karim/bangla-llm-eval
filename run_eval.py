"""Run every item against one model through an OpenAI- or Anthropic-compatible endpoint."""
import asyncio, json, os, sys, time
from pathlib import Path
import httpx

provider, model, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
opts = set(sys.argv[4:])
base, key = os.environ["BN_EVAL_BASE_URL"].rstrip("/"), os.environ["BN_EVAL_API_KEY"]
items = [json.loads(l) for l in Path("items.jsonl").read_text(encoding="utf-8").splitlines()]
done = set()
if out.exists():
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}

async def call(client, item):
    if provider == "anthropic":
        body = {"model": model, "max_tokens": 512, "messages": [{"role": "user", "content": item["prompt"]}]}
        if "notemp" not in opts:
            body["temperature"] = 0
        r = await client.post(base + "/messages", json=body,
                              headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
        r.raise_for_status(); d = r.json()
        text = "".join(b.get("text", "") for b in d["content"] if b.get("type") == "text")
        return text, d.get("usage", {})
    body = {"model": model, "messages": [{"role": "user", "content": item["prompt"]}]}
    body["max_completion_tokens" if "mct" in opts else "max_tokens"] = 2048 if "mct" in opts else 512
    if "notemp" not in opts:
        body["temperature"] = 0
    r = await client.post(base + "/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status(); d = r.json()
    return d["choices"][0]["message"]["content"] or "", d.get("usage", {})

async def main():
    sem = asyncio.Semaphore(6)
    async with httpx.AsyncClient(timeout=120) as client:
        async def one(item):
            async with sem:
                for attempt in range(3):
                    t = time.time()
                    try:
                        text, usage = await call(client, item)
                        rec = {"id": item["id"], "output": text, "usage": usage, "latency_s": round(time.time() - t, 2)}
                        break
                    except Exception as e:  # recorded, never echoes headers
                        rec = {"id": item["id"], "error": type(e).__name__ + (f" {e.response.status_code}" if isinstance(e, httpx.HTTPStatusError) else "")}
                        await asyncio.sleep(2 * (attempt + 1))
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        await asyncio.gather(*(one(i) for i in items if i["id"] not in done))
asyncio.run(main())
print("DONE", out)
