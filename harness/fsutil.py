"""File writes that only need the *directory* to be writable.

The jury runs the container with --cap-drop ALL, so root cannot override file
permissions: a 0644 file owned by another user cannot be opened for writing,
but it can be replaced by a new file in a writable directory (os.replace).
"""
import os
import stat
import tempfile


def atomic_write(path, text):
    d = os.path.dirname(os.path.abspath(path))
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode) | 0o644
    except OSError:
        mode = 0o644
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, mode)  # mkstemp creates 0600; keep the file readable for the checker
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
