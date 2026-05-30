"""Sandbox execution backends.

Five backends: local (subprocess), Docker, SSH — all implement execute().
Missing optional libraries are handled with ImportError guards and clear
instructions rather than crashing.

Usage:
    mgr = SandboxManager(config)
    result = mgr.execute("print('hello')", language="python")
    # {"success": True, "stdout": "hello\\n", "stderr": "", "backend": "local"}
"""
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Optional

from ..core.logger import get_logger

log = get_logger(__name__)

# ── Optional deps ─────────────────────────────────────────────────────────

_DOCKER_AVAILABLE = False
try:
    import docker as docker_sdk  # type: ignore[import]
    _DOCKER_AVAILABLE = True
except ImportError:
    pass

_PARAMIKO_AVAILABLE = False
try:
    import paramiko  # type: ignore[import]
    _PARAMIKO_AVAILABLE = True
except ImportError:
    pass


# ── Result helper ─────────────────────────────────────────────────────────

def _result(success: bool, stdout: str = "", stderr: str = "", backend: str = "?") -> dict:
    return {"success": success, "stdout": stdout, "stderr": stderr, "backend": backend}


# ── Local backend ─────────────────────────────────────────────────────────

class LocalBackend:
    """Execute code in a subprocess on the host machine."""

    name = "local"

    def execute(self, code: str, language: str = "python", timeout: int = 30) -> dict:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=_ext(language), delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp = f.name

        try:
            proc = subprocess.run(
                _cmd(language, tmp),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return _result(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                backend=self.name,
            )
        except subprocess.TimeoutExpired:
            return _result(False, stderr=f"Timed out after {timeout}s", backend=self.name)
        except Exception as exc:
            log.exception("[sandbox/local] Execution error")
            return _result(False, stderr=str(exc), backend=self.name)
        finally:
            Path(tmp).unlink(missing_ok=True)


# ── Docker backend ─────────────────────────────────────────────────────────

class DockerBackend:
    """Execute code inside a Docker container for strong isolation."""

    name = "docker"

    def __init__(
        self,
        image: str = "python:3.11-slim",
        memory_limit: str = "256m",
        cpu_quota: int = 50000,
        network_disabled: bool = True,
    ):
        self._image = image
        self._memory = memory_limit
        self._cpu_quota = cpu_quota
        self._net_disabled = network_disabled
        self._client = None

    def _get_client(self):
        if not _DOCKER_AVAILABLE:
            raise RuntimeError(
                "docker SDK not installed — install with: pip install docker"
            )
        if self._client is None:
            self._client = docker_sdk.from_env()
        return self._client

    def execute(self, code: str, language: str = "python", timeout: int = 30) -> dict:
        try:
            client = self._get_client()
        except RuntimeError as exc:
            log.error("[sandbox/docker] %s", exc)
            return _result(False, stderr=str(exc), backend=self.name)

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / f"code{_ext(language)}"
            src.write_text(code, encoding="utf-8")
            container_src = f"/code/code{_ext(language)}"

            try:
                output = client.containers.run(
                    self._image,
                    command=_cmd(language, container_src),
                    volumes={tmpdir: {"bind": "/code", "mode": "ro"}},
                    mem_limit=self._memory,
                    cpu_quota=self._cpu_quota,
                    network_disabled=self._net_disabled,
                    remove=True,
                    detach=False,
                    stdout=True,
                    stderr=True,
                    timeout=timeout,
                )
                text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)
                return _result(True, stdout=text, backend=self.name)
            except Exception as exc:
                log.exception("[sandbox/docker] Container execution error")
                return _result(False, stderr=str(exc), backend=self.name)


# ── SSH backend ─────────────────────────────────────────────────────────────

class SSHBackend:
    """Execute code on a remote host via SSH + SFTP."""

    name = "ssh"

    def __init__(
        self,
        host: str,
        port: int = 22,
        username: str = "",
        key_path: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self._host = host
        self._port = port
        self._username = username
        self._key_path = key_path
        self._password = password
        self._lock = threading.Lock()

    def execute(self, code: str, language: str = "python", timeout: int = 30) -> dict:
        if not _PARAMIKO_AVAILABLE:
            msg = "paramiko not installed — install with: pip install paramiko"
            log.error("[sandbox/ssh] %s", msg)
            return _result(False, stderr=msg, backend=self.name)

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs: dict = dict(hostname=self._host, port=self._port,
                            username=self._username, timeout=10)
        if self._key_path:
            kwargs["key_filename"] = self._key_path
        if self._password:
            kwargs["password"] = self._password

        remote_path = f"/tmp/hermes_{id(code)}{_ext(language)}"
        try:
            with self._lock:
                client.connect(**kwargs)

            sftp = client.open_sftp()
            with sftp.open(remote_path, "w") as f:
                f.write(code)
            sftp.close()

            _, stdout, stderr = client.exec_command(
                " ".join(_cmd(language, remote_path)), timeout=timeout
            )
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            rc = stdout.channel.recv_exit_status()
            client.exec_command(f"rm -f {remote_path}")
            client.close()
            return _result(rc == 0, stdout=out, stderr=err, backend=self.name)

        except Exception as exc:
            log.exception("[sandbox/ssh] Remote execution error")
            return _result(False, stderr=str(exc), backend=self.name)


# ── Manager ──────────────────────────────────────────────────────────────────

class SandboxManager:
    """Routes code execution to the configured backend."""

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        backend_name = cfg.get("backend", "local")

        if backend_name == "docker":
            self._backend = DockerBackend(
                image=cfg.get("docker_image", "python:3.11-slim"),
                memory_limit=cfg.get("memory_limit", "256m"),
                network_disabled=cfg.get("network_disabled", True),
            )
        elif backend_name == "ssh":
            self._backend = SSHBackend(
                host=cfg.get("host", "localhost"),
                port=int(cfg.get("port", 22)),
                username=cfg.get("username", ""),
                key_path=cfg.get("key_path"),
                password=cfg.get("password"),
            )
        else:
            self._backend = LocalBackend()

        log.info("[sandbox] Using backend: %s", self._backend.name)

    def execute(self, code: str, language: str = "python", timeout: int = 30) -> dict:
        return self._backend.execute(code, language, timeout)

    @property
    def backend_name(self) -> str:
        return self._backend.name


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ext(language: str) -> str:
    return {
        "python": ".py", "javascript": ".js", "typescript": ".ts",
        "bash": ".sh", "shell": ".sh",
    }.get(language.lower(), ".py")


def _cmd(language: str, path: str) -> list[str]:
    return {
        "python": ["python3", path],
        "javascript": ["node", path],
        "bash": ["bash", path],
        "shell": ["bash", path],
    }.get(language.lower(), ["python3", path])
