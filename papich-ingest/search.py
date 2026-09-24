#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
from src.config import load_config
from src.metadata_store import MetadataStore
from src.utils import database_lock, timestamp


def semantic_search(query, config, top_k=None):
    """Reusable retrieval function for a future chatbot; returns scores and full metadata."""
    if not query.strip():
        raise ValueError('Query cannot be empty')
    limit = config.top_k if top_k is None else top_k
    if limit < 1:
        raise ValueError('top_k must be positive')
    if not (config.path('db') / 'metadata.sqlite').exists():
        return []
    with database_lock(config.path('db') / 'pipeline.lock'):
        metadata = MetadataStore(config.path('db') / 'metadata.sqlite')
        try:
            metadata.ensure_index_profile(config.fingerprint('index'))
            ids = [r[0] for r in metadata.db.execute(
                'SELECT id FROM videos WHERE status="completed" AND indexed=1')]
        finally:
            metadata.close()
        if not ids:
            return []
        from src.embeddings import Embedder
        from src.vector_store import VectorStore
        embedder = Embedder(config)
        store = VectorStore(config.path('db') / 'vectors', embedder.dimension)
        try:
            return [{'score': hit.score, **hit.payload} for hit in
                    store.search(embedder.encode([query], query=True)[0], limit, ids)]
        finally:
            store.close()


def main():
    parser = argparse.ArgumentParser(description='Search the local video knowledge base')
    parser.add_argument('query')
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--json', action='store_true', help='Machine-readable results')
    args = parser.parse_args()
    try:
        hits = semantic_search(args.query, load_config(args.config), args.top_k)
        if args.json:
            print(json.dumps(hits, ensure_ascii=False, indent=2))
        elif not hits:
            print('No indexed chunks. Run ingest.py without --skip-embeddings first.')
        else:
            for hit in hits:
                print(f"[{hit['score']:.4f}] {hit['title']}\n"
                      f"{timestamp(hit['start'])}–{timestamp(hit['end'])}\n"
                      f"{hit['timestamp_url']}\n{hit['text']}\n")
        return 0
    except Exception as exc:
        print(f'Search failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
