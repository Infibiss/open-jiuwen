"""Run model-written Python in a subprocess confined to the task directory."""
import os
import re
import subprocess
import sys
import tempfile

_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    blocks = _CODE.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    t = (text or "").strip()
    if t.startswith(("import ", "from ", "print(", "x =", "s =", "stock")):
        return t
    return None


def _limits():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    except Exception:  # noqa: BLE001 - not fatal (e.g. macOS AS limit)
        pass


def run_python(code, cwd, timeout=30):
    """Returns (ok, stdout_last_line, full_output)."""
    fd, path = tempfile.mkstemp(suffix=".py", prefix="snip_", dir="/tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        p = subprocess.run([sys.executable, "-I", path], cwd=cwd, capture_output=True, text=True,
                           timeout=timeout, preexec_fn=_limits if os.name == "posix" else None)
        out = p.stdout.strip()
        last = out.splitlines()[-1].strip() if out else ""
        return p.returncode == 0 and bool(last), last, (p.stdout + p.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
