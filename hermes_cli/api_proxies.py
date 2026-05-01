"""Proxy routers for nanobot agent core and Docker Engine API.

Mounted under /api/nanobot/* and /api/docker/* in the hermes web server.
"""
import asyncio
import os
import urllib.error
import urllib.request

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response


# ---------------------------------------------------------------------------
# Configuration — read from environment so Docker can inject NANOBOT_API_URL
# ---------------------------------------------------------------------------

# nanobot agent-core — container name on the Docker network, or host machine
NANOBOT_URL = os.environ.get("NANOBOT_API_URL", "http://localhost:8900")

# Docker Engine API — TCP Docker daemon (host machine)
DOCKER_URL = os.environ.get("DOCKER_HOST_URL", "http://localhost:2375")


# ---------------------------------------------------------------------------
# nanobot agent core proxy
# ---------------------------------------------------------------------------

nanobot_router = APIRouter(prefix="/nanobot", tags=["nanobot"])


@nanobot_router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def nanobot_proxy(path: str, request: Request):
    """Proxy all requests to the nanobot agent-core OpenAI-compatible API."""
    url = f"{NANOBOT_URL}/{path}"
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    req = urllib.request.Request(url, data=body, headers=headers, method=request.method)
    try:
        resp = await _forward(req)
    except urllib.error.URLError as e:
        raise HTTPException(status_code=502, detail=f"nanobot unreachable: {e}")
    return Response(
        content=resp["body"],
        status_code=resp["code"],
        headers=resp["headers"],
    )


# ---------------------------------------------------------------------------
# Docker Engine API proxy
# ---------------------------------------------------------------------------

docker_router = APIRouter(prefix="/docker", tags=["docker"])


@docker_router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def docker_proxy(path: str, request: Request):
    """Proxy Docker Engine API requests from the React dashboard.

    The React app runs in a browser and cannot directly reach the Docker
    Unix socket, so all Docker API calls are forwarded through the hermes
    web server which has access to the host Docker daemon.
    """
    url = f"{DOCKER_URL}/{path}"
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)
    req = urllib.request.Request(url, data=body, headers=headers, method=request.method)
    try:
        resp = await _forward(req)
    except urllib.error.URLError as e:
        raise HTTPException(status_code=502, detail=f"Docker unreachable: {e}")
    return Response(
        content=resp["body"],
        status_code=resp["code"],
        headers=resp["headers"],
    )


# ---------------------------------------------------------------------------
# Shared forwarding helper
# ---------------------------------------------------------------------------

async def _forward(req: urllib.request.Request) -> dict:
    """Execute a blocking urllib request in a thread pool and return response."""
    def _do() -> dict:
        try:
            raw_resp = urllib.request.urlopen(req, timeout=30)
            return {
                "code": raw_resp.getcode(),
                "headers": {k: v for k, v in raw_resp.headers.items()},
                "body": raw_resp.read(),
            }
        except urllib.error.HTTPError as e:
            return {
                "code": e.code,
                "headers": {k: v for k, v in e.headers.items()},
                "body": e.read(),
            }

    return await asyncio.get_running_loop().run_in_executor(None, _do)
