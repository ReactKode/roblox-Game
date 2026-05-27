"""Sandboxed Python code execution."""
import subprocess
import sys
import tempfile
import os
from pathlib import Path


def run_python(code: str, timeout: int = 30, working_dir: str = None) -> dict:
    """Execute Python code in a subprocess and return stdout/stderr/success."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=working_dir or Path.home(),
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:2000],
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"Timeout after {timeout}s", "return_code": -1}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "return_code": -1}
    finally:
        os.unlink(tmp_path)


def run_shell(command: str, timeout: int = 30) -> dict:
    """Execute a shell command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:2000],
            "return_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": f"Timeout after {timeout}s", "return_code": -1}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "return_code": -1}
