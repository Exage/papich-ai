from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os
import tempfile


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def write_json(path, obj):
    atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_jsonl(path, rows):
    atomic_text(path, ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n'
                              for r in rows))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def digest(path):
    with Path(path).open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


@contextmanager
def database_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This database is busy. Wait for ingestion/search to finish.')
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def timestamp(seconds):
    value = max(0, int(seconds))
    return f'{value // 3600:02d}:{value // 60 % 60:02d}:{value % 60:02d}'
