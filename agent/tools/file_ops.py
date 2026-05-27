"""File system operations tool."""
import os
from pathlib import Path


def read_file(path: str, max_chars: int = 10000) -> str:
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return f"Error: File not found: {path}"
        text = p.read_text(encoding="utf-8", errors="replace")
        return text[:max_chars]
    except Exception as e:
        return f"Error reading {path}: {e}"


def write_file(path: str, content: str, append: bool = False) -> str:
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        if append:
            with p.open("a", encoding="utf-8") as f:
                f.write(content)
        else:
            p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"Error writing {path}: {e}"


def list_files(directory: str = ".", pattern: str = "*") -> str:
    try:
        p = Path(directory).expanduser()
        if not p.exists():
            return f"Error: Directory not found: {directory}"
        items = []
        for item in sorted(p.glob(pattern)):
            kind = "dir" if item.is_dir() else "file"
            size = item.stat().st_size if item.is_file() else 0
            items.append(f"[{kind}] {item.name} ({size} bytes)" if kind == "file" else f"[{kind}] {item.name}/")
        return "\n".join(items) if items else "Empty directory."
    except Exception as e:
        return f"Error listing {directory}: {e}"


def create_directory(path: str) -> str:
    try:
        Path(path).expanduser().mkdir(parents=True, exist_ok=True)
        return f"Directory created: {path}"
    except Exception as e:
        return f"Error creating directory {path}: {e}"
