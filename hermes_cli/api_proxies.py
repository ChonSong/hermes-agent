"""Proxy routers for nanobot agent core and Docker Engine API.

Mounted under /api/nanobot/* and /api/docker/* in the hermes web server.
"""
import asyncio
import os
import subprocess
import urllib.error
import urllib.request

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NANOBOT_URL = os.environ.get("NANOBOT_API_URL", "http://localhost:8900")

# Docker Engine API via Unix socket (mount /var/run/docker.sock into the
# dashboard container) or TCP daemon (DOCKER_HOST_URL=http://host:2375).
# Default: Unix socket. Set DOCKER_HOST_URL to override.
DOCKER_SOCKET = os.environ.get("DOCKER_SOCKET", "/var/run/docker.sock")
DOCKER_TCP_URL = os.environ.get("DOCKER_HOST_URL", "")


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
        resp = await _forward_http(req)
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

    Browser → hermes web server → Docker Engine API (Unix socket or TCP).
    Docker API paths forwarded as-is: /containers/json → /containers/json.
    """
    if DOCKER_TCP_URL:
        # TCP daemon
        url = f"{DOCKER_TCP_URL}/{path}"
        body = await request.body()
        headers = dict(request.headers)
        headers.pop("host", None)
        req = urllib.request.Request(url, data=body, headers=headers, method=request.method)
        try:
            resp = await _forward_http(req)
        except urllib.error.URLError as e:
            raise HTTPException(status_code=502, detail=f"Docker unreachable: {e}")
        return Response(
            content=resp["body"],
            status_code=resp["code"],
            headers=resp["headers"],
        )
    else:
        # Unix socket — use docker CLI which handles socket transport
        docker_path = _build_docker_command(path, request)
        try:
            resp = await _run_docker_cli(docker_path, request)
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=e.returncode or 500, detail=f"Docker CLI error: {e.stderr}")
        return resp


def _build_docker_command(path: str, request: Request) -> str:
    """Map Docker API path+method to a docker CLI command."""
    # Strip leading slash
    path = path.lstrip("/")
    return path


async def _run_docker_cli(path: str, request: Request) -> Response:
    """Execute docker CLI command for Docker Engine API via Unix socket."""
    method = request.method
    body = await request.body()

    def _do() -> tuple[int, bytes, dict]:
        # Build docker CLI args from the API path
        # e.g. "containers/json?all=true" → docker ps -a --format json
        # e.g. "containers/{id}/start" → docker start {id}
        # e.g. "containers/{id}" DELETE → docker rm {id}
        # e.g. "containers/{id}/logs" → docker logs {id}
        parts = path.split("/")
        args = ["docker"]

        if parts[0] == "containers":
            if len(parts) == 2 or (len(parts) == 3 and parts[2] == "json"):
                # GET /containers/json → docker ps -a
                args += ["ps", "-a", "--format=json"]
                if "all=true" not in path:
                    args.append("--all")
            elif len(parts) >= 3:
                container_id = parts[1]
                action = parts[2] if len(parts) > 2 else None
                if method == "DELETE":
                    args = ["docker", "rm", "-f", container_id]
                elif method == "POST":
                    if action == "start":
                        args = ["docker", "start", container_id]
                    elif action == "stop":
                        args = ["docker", "stop", container_id]
                    elif action == "restart":
                        args = ["docker", "restart", container_id]
                    elif action == "logs":
                        args = ["docker", "logs", container_id]
                    elif action == "inspect":
                        args = ["docker", "inspect", container_id]
                    else:
                        raise ValueError(f"Unknown Docker action: {action}")
                elif method == "GET":
                    if action == "logs":
                        args = ["docker", "logs", container_id]
                    elif action == "inspect":
                        args = ["docker", "inspect", container_id]
                    elif action == "stats":
                        args = ["docker", "stats", container_id, "--no-stream", "--format=json"]
                    else:
                        args = ["docker", "inspect", container_id]
        elif parts[0] == "images":
            if len(parts) == 2 or parts[1] == "json":
                args = ["docker", "images", "--format=json"]
        elif parts[0] == "version":
            args = ["docker", "version", "--format=json"]
        elif parts[0] == "info":
            args = ["docker", "info", "--format=json"]
        elif parts[0] == "system":
            if len(parts) > 1 and parts[1] == "df":
                args = ["docker", "system", "df", "--format=json"]
        else:
            raise ValueError(f"Unsupported Docker API path: {path}")

        proc = subprocess.run(
            args,
            capture_output=True,
            input=body if body else None,
            timeout=30,
        )
        content_type = "application/json"
        return proc.returncode, proc.stdout, {"Content-Type": content_type}

    loop = asyncio.get_event_loop()
    code, body_bytes, headers = await loop.run_in_executor(None, _do)
    return Response(content=body_bytes, status_code=code, headers=headers)


# ---------------------------------------------------------------------------
# Shared forwarding helper
# ---------------------------------------------------------------------------

async def _forward_http(req: urllib.request.Request) -> dict:
    """Execute a blocking urllib request in a thread pool."""
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

# --------------------------------------------------------------------------
# Docker Hub search
# --------------------------------------------------------------------------

misc_router = APIRouter(tags=["misc"])


@misc_router.get("/docker/search")
async def docker_hub_search(q: str = "", limit: int = 20):
    """Search Docker Hub via the v2 API (no auth required)."""
    import urllib.parse, urllib.request, json
    if not q:
        return []
    url = f"https://hub.docker.com/v2/search/repositories/?query={urllib.parse.quote(q)}&page_size={limit}"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get("results", [])
    except Exception:
        return []


@misc_router.get("/tunnels")
async def list_tunnels():
    """List active Cloudflare tunnels via cloudflared CLI."""
    import subprocess, json
    try:
        result = subprocess.run(
            ["cloudflared", "tunnel", "list", "--output", "json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            tunnels = json.loads(result.stdout)
            return [
                {
                    "name": t.get("name", t.get("id", "")),
                    "url": (t.get("connections") or [{}])[0].get("uri", ""),
                    "status": t.get("status", {}).get("string", "active"),
                }
                for t in tunnels
            ]
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["cloudflared", "tunnel", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            tunnels = []
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 3:
                    tunnels.append({"name": parts[0], "url": parts[1], "status": parts[2]})
            return tunnels
    except Exception:
        pass
    return []


# --------------------------------------------------------------------------
# Filesystem browser
# --------------------------------------------------------------------------


files_router = APIRouter(prefix="/files", tags=["files"])


@files_router.get("/list")
async def list_directory(path: str = "/root"):
    """List a directory's contents."""
    import subprocess
    try:
        result = subprocess.run(
            ["ls", "-la", "--time-style=long-iso", path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        lines = result.stdout.strip().split("\n")
        entries = []
        for line in lines[1:]:
            if not line.strip():
                continue
            parts = line.split(None, 7)
            if len(parts) < 8:
                continue
            mode = parts[0]
            size = parts[4]
            month, day, time_or_year = parts[5], parts[6], parts[7]
            name = parts[7]
            if " -> " in name:
                name = name.split(" -> ")[0]
            full_path = path.rstrip("/") + "/" + name if path != "/" else "/" + name
            entries.append({
                "name": name,
                "path": full_path,
                "type": "directory" if mode.startswith("d") else "file",
                "size": int(size) if size.isdigit() else 0,
                "modTime": f"{month} {day} {time_or_year}",
            })
        return entries
    except Exception:
        return []


@files_router.get("/read")
async def read_file(path: str):
    """Return text content of a file."""
    from fastapi.responses import PlainTextResponse
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(1024 * 512)
        return PlainTextResponse(content)
    except Exception as e:
        return PlainTextResponse(str(e), status_code=400)


@files_router.post("/mkdir")
async def make_directory(request: Request):
    """Create a directory."""
    import subprocess
    body = await request.json()
    dir_path = body.get("path", "/root")
    name = body.get("name", "")
    full = dir_path.rstrip("/") + "/" + name
    result = subprocess.run(["mkdir", "-p", full], capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr or "mkdir failed"}
    return {"ok": True, "path": full}


@files_router.delete("/delete")
async def delete_path(request: Request):
    """Delete a file or directory."""
    import subprocess
    body = await request.json()
    target = body.get("path", "")
    if not target:
        return {"error": "No path provided"}
    if target in ["/", "/root", "/home", "/var", "/etc", "/usr"]:
        return {"error": "Refusing to delete protected path"}
    result = subprocess.run(["rm", "-rf", target], capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": result.stderr or "Delete failed"}
    return {"ok": True}


@files_router.get("/download")
async def download_file(path: str):
    """Serve a file for download."""
    from fastapi.responses import FileResponse
    from pathlib import Path
    safe_path = Path(path).resolve()
    if not safe_path.exists():
        return {"error": "File not found"}
    return FileResponse(path=safe_path, filename=safe_path.name)


@files_router.post("/upload")
async def upload_file(request: Request):
    """Upload a file to a directory."""
    import os
    form = await request.form()
    file = form.get("file")
    dest_dir = form.get("path", "/root")
    if not file:
        return {"error": "No file provided"}
    filename = file.filename or "uploaded"
    dest = os.path.join(dest_dir, filename)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    return {"ok": True, "path": dest}
