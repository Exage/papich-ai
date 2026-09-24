import json
import sqlite3
from .utils import digest

STAGES = ['discovered', 'downloaded', 'transcribed', 'chunked', 'embedded', 'completed']


class MetadataStore:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS videos (
            id TEXT PRIMARY KEY, metadata TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'discovered',
            last_success TEXT NOT NULL DEFAULT 'discovered',
            error TEXT, indexed INTEGER NOT NULL DEFAULT 0,
            asr_key TEXT, chunks_key TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS artifacts (
            video_id TEXT, name TEXT, path TEXT, sha256 TEXT,
            PRIMARY KEY(video_id, name));
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, video_id TEXT NOT NULL, text TEXT NOT NULL,
            duplicate_of TEXT, duplicate_score REAL);
        CREATE INDEX IF NOT EXISTS chunks_video ON chunks(video_id);
        CREATE TABLE IF NOT EXISTS errors (
            url TEXT, video_id TEXT, message TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
        ''')

    def discover(self, video):
        with self.db:
            self.db.execute('INSERT INTO videos(id,metadata) VALUES (?,?) '
                            'ON CONFLICT(id) DO UPDATE SET metadata=excluded.metadata',
                            (video.id, json.dumps(video.to_dict(), ensure_ascii=False)))

    def get(self, key):
        row = self.db.execute('SELECT * FROM videos WHERE id=?', (key,)).fetchone()
        return dict(row) if row else None

    def advance(self, key, stage, indexed=None):
        if stage not in STAGES:
            raise ValueError(f'Unknown stage: {stage}')
        row = self.get(key)
        if row is None or STAGES.index(stage) < STAGES.index(row['last_success']):
            raise ValueError('Use rewind() to invalidate downstream stages')
        with self.db:
            self.db.execute('UPDATE videos SET status=?,last_success=?,error=NULL,'
                            'indexed=?,updated_at=CURRENT_TIMESTAMP WHERE id=?',
                            (stage, stage, row['indexed'] if indexed is None else indexed, key))

    def rewind(self, key, stage='discovered'):
        if stage not in STAGES:
            raise ValueError(stage)
        with self.db:
            self.db.execute('UPDATE videos SET status=?,last_success=?,indexed=0,error=NULL '
                            'WHERE id=?', (stage, stage, key))

    def fail(self, key, url, error):
        with self.db:
            self.db.execute('UPDATE videos SET status="failed",error=?, '
                            'updated_at=CURRENT_TIMESTAMP WHERE id=?', (str(error), key))
            self.db.execute('INSERT INTO errors(url,video_id,message) VALUES (?,?,?)',
                            (url, key, str(error)))

    def artifact(self, key, name, path):
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO artifacts VALUES (?,?,?,?)',
                            (key, name, str(path), digest(path)))

    def valid_artifact(self, key, name):
        row = self.db.execute('SELECT path,sha256 FROM artifacts WHERE video_id=? AND name=?',
                              (key, name)).fetchone()
        try:
            return bool(row and digest(row['path']) == row['sha256'])
        except OSError:
            return False

    def invalidate(self, key, names):
        with self.db:
            self.db.executemany('DELETE FROM artifacts WHERE video_id=? AND name=?',
                                [(key, name) for name in names])

    def set_keys(self, key, asr_key, chunks_key):
        with self.db:
            self.db.execute('UPDATE videos SET asr_key=?,chunks_key=? WHERE id=?',
                            (asr_key, chunks_key, key))

    def replace_chunks(self, key, rows):
        with self.db:
            self.db.execute('DELETE FROM chunks WHERE video_id=?', (key,))
            self.db.executemany('INSERT INTO chunks VALUES (?,?,?,?,?)',
                                [(r['id'], key, r['text'], r.get('duplicate_of'),
                                  r.get('duplicate_score')) for r in rows])

    def candidates(self, exclude_video):
        return self.db.execute('SELECT id,text FROM chunks WHERE video_id<>? ORDER BY id',
                               (exclude_video,))

    def ensure_index_profile(self, profile):
        row = self.db.execute('SELECT value FROM settings WHERE key="index_profile"').fetchone()
        if row and row[0] != profile:
            raise ValueError('Embedding model/prefix changed. Use a new data_dir and reindex; '
                             'vectors from different models cannot share this database.')
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO settings VALUES ("index_profile",?)', (profile,))

    def close(self):
        self.db.close()
