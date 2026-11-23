#!/usr/bin/env python3
"""Automated Disk-Session Proxy for Strata (OpenAI API Compatible).

Transparently saves and restores KV cache states to/from fast NVMe disk
between requests, keeping host RAM 100% free for model expert weights.

Usage:
    .venv/bin/python proxy.py --port 8000 --backend http://127.0.0.1:8080
"""

import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager
import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("strata-proxy")

ROOT = Path(__file__).resolve().parent
DEFAULT_SESSIONS_DIR = Path("/var/kvcache/strata-rdna3")


class ProxyState:
    backend_url: str = "http://127.0.0.1:8080"
    sessions_dir: Path = DEFAULT_SESSIONS_DIR
    active_session: Optional[str] = None
    lock: asyncio.Lock = asyncio.Lock()
    client: Optional[httpx.AsyncClient] = None


state = ProxyState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.sessions_dir.mkdir(parents=True, exist_ok=True)
    state.client = httpx.AsyncClient(base_url=state.backend_url, timeout=httpx.Timeout(3600.0, connect=10.0))
    logger.info("Strata Proxy connected to backend %s. Sessions dir: %s", state.backend_url, state.sessions_dir)
    yield
    if state.active_session:
        try:
            await save_slot(state.active_session)
        except Exception as e:
            logger.warning("Failed to save final session on shutdown: %s", e)
    if state.client:
        await state.client.aclose()


app = FastAPI(title="Strata Disk Session Proxy", lifespan=lifespan)


def get_session_id(request: Request, body: dict) -> str:
    """Extract or hash a persistent session identifier."""
    # 1. Explicit headers
    for h in ("x-session-id", "session-id", "x-conversation-id", "conversation-id"):
        val = request.headers.get(h)
        if val:
            return hashlib.sha256(val.encode()).hexdigest()[:16]

    # 2. 'user' field in payload
    if body.get("user") and isinstance(body["user"], str):
        return hashlib.sha256(body["user"].encode()).hexdigest()[:16]

    # 3. Deterministic hash of system prompt and first user turn
    messages = body.get("messages", [])
    if messages:
        prefix_parts = []
        for m in messages[:2]:
            prefix_parts.append(f"{m.get('role')}:{m.get('content')}")
        sig = "|".join(prefix_parts)
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

    return "default_session"


async def save_slot(session_id: str):
    """Save active slot 0 to disk."""
    filename = f"{session_id}.bin"
    logger.info("Persisting session %s to disk (sessions/%s)...", session_id, filename)
    try:
        resp = await state.client.post("/slots/0", params={"action": "save"}, json={"filename": filename})
        if resp.status_code == 200:
            data = resp.json()
            logger.info("Session %s saved successfully (%d tokens, %s bytes in %d ms)",
                        session_id, data.get("n_saved", 0), data.get("n_written", 0),
                        data.get("timings", {}).get("save_ms", 0))
        else:
            logger.warning("Failed to save session %s: %s %s", session_id, resp.status_code, resp.text)
    except Exception as e:
        logger.error("Error saving session %s: %s", session_id, e)


async def restore_slot(session_id: str) -> bool:
    """Restore slot 0 from disk."""
    filename = f"{session_id}.bin"
    file_path = state.sessions_dir / filename
    if not file_path.is_file():
        return False

    logger.info("Restoring session %s from NVMe disk (%s)...", session_id, filename)
    try:
        resp = await state.client.post("/slots/0", params={"action": "restore"}, json={"filename": filename})
        if resp.status_code == 200:
            data = resp.json()
            logger.info("Session %s restored successfully (%d tokens in %d ms)",
                        session_id, data.get("n_restored", 0), data.get("timings", {}).get("restore_ms", 0))
            return True
        else:
            logger.warning("Restore refused for session %s: %s %s", session_id, resp.status_code, resp.text)
            return False
    except Exception as e:
        logger.error("Error restoring session %s: %s", session_id, e)
        return False


async def switch_session(target_session: str):
    """Safely swap active slot on Strata."""
    async with state.lock:
        if state.active_session == target_session:
            return

        # Save existing active session before swapping
        if state.active_session:
            await save_slot(state.active_session)

        # Restore target if exists
        restored = await restore_slot(target_session)
        state.active_session = target_session
        if not restored:
            logger.info("Started fresh session %s on slot 0", target_session)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    session_id = get_session_id(request, body)
    await switch_session(session_id)

    stream = body.get("stream", False)
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}

    if stream:
        async def stream_generator():
            try:
                async with state.client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            finally:
                # Save state after turn finishes
                asyncio.create_task(save_slot(session_id))

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        resp = await state.client.post("/v1/chat/completions", json=body, headers=headers)
        asyncio.create_task(save_slot(session_id))
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def catch_all(request: Request, path: str):
    """Directly forward all other endpoints (models, health, status, etc.)."""
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    body = await request.body()
    resp = await state.client.request(
        method=request.method,
        url=f"/{path}",
        headers=headers,
        content=body,
        params=request.query_params
    )
    return Response(content=resp.content, status_code=resp.status_code, media_type=resp.headers.get("content-type"))


def main():
    parser = argparse.ArgumentParser(description="Strata Automated Disk-Session Proxy")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind proxy")
    parser.add_argument("--port", type=int, default=8000, help="Client-facing proxy port (default: 8000)")
    parser.add_argument("--backend", default="http://127.0.0.1:8080", help="Strata backend URL (default: http://127.0.0.1:8080)")
    parser.add_argument("--sessions-dir", default=str(DEFAULT_SESSIONS_DIR), help="Directory for saved session files")
    args = parser.parse_args()

    state.backend_url = args.backend
    state.sessions_dir = Path(args.sessions_dir).resolve()

    print(f"\n=======================================================")
    print(f"🚀 Strata Automated Disk-Session Proxy")
    print(f"   Proxy Endpoint:  http://{args.host}:{args.port}/v1")
    print(f"   Backend Strata:  {args.backend}")
    print(f"   NVMe Sessions:   {state.sessions_dir}")
    print(f"=======================================================\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
