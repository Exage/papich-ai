"""Heavy stages run sequentially in separate OS processes to release unified RAM."""
from dataclasses import asdict
from pathlib import Path
import json
import logging
import subprocess
import sys
from .config import Config
from .utils import write_json, read_json, read_jsonl, write_jsonl


def run_worker(stage, config, **kwargs):
    # JSON on stdin avoids quoting issues and temporary job files.
    job = {'stage': stage, 'config': asdict(config), **kwargs}
    result = subprocess.run([sys.executable, '-m', 'src.worker'],
                            input=json.dumps(job), text=True, cwd=Path(__file__).resolve().parent.parent)
    if result.returncode:
        raise RuntimeError(f'{stage} worker failed (exit {result.returncode}); see data/logs/ingest.log')


def execute(job):
    config = Config(**job['config'])
    stage = job['stage']
    if stage == 'transcribe':
        from .transcriber import transcribe
        transcribe(job['audio'], job['raw'], job['video'], config)
    elif stage == 'diarize':
        from .diarization import diarize
        diarize(job['audio'], job['raw'], job['diarized'], config)
    elif stage == 'chunk':
        from .cleaner import clean_segments
        from .chunker import chunk_segments
        from .embeddings import token_counter
        raw = read_json(job.get('diarized') or job['raw'])
        segments = clean_segments(raw['segments'])
        write_json(job['clean'], {'video': job['video'], 'segments': segments})
        chunks = chunk_segments(segments, job['video'], config, token_counter(config)) if segments else []
        write_jsonl(job['chunks'], chunks)
    elif stage == 'embed':
        from .embeddings import Embedder
        from .vector_store import VectorStore
        chunks = read_jsonl(job['chunks'])
        if not chunks:
            return
        embedder = Embedder(config)
        store = VectorStore(config.path('db') / 'vectors', embedder.dimension)
        try:
            store.delete_video(job['video']['id'])
            size = config.embedding_batch_size
            for start in range(0, len(chunks), size):
                batch = chunks[start:start + size]
                vectors = embedder.encode([c['text'] for c in batch])
                store.upsert(batch, vectors)
                logging.info('Embedded %d/%d chunks', min(start + size, len(chunks)), len(chunks))
        finally:
            store.close()
    else:
        raise ValueError(f'Unknown worker stage: {stage}')


if __name__ == '__main__':
    job = json.load(sys.stdin)
    config = Config(**job['config'])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(
                            config.path('logs') / 'ingest.log', encoding='utf-8')])
    logging.getLogger('httpx').setLevel(logging.WARNING)
    try:
        execute(job)
    except Exception:
        logging.exception('%s failed', job['stage'])
        sys.exit(1)
