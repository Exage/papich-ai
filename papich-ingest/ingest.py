#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import sys
from src.config import load_config
from src.downloader import expand_url
from src.pipeline import Pipeline
from src.urls import video_id
from src.utils import database_lock


def main():
    parser = argparse.ArgumentParser(description='Local YouTube → MLX Whisper → RAG knowledge base')
    parser.add_argument('file', nargs='?', help='UTF-8 file with one URL per line')
    parser.add_argument('--url', action='append', default=[], help='Video URL (repeatable)')
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--model', help='tiny/base/small/medium/large-v3/turbo or MLX model path/repo')
    parser.add_argument('--force', action='store_true', help='Re-download and reprocess supplied videos')
    parser.add_argument('--skip-embeddings', action='store_true', help='Stop at chunked; resume indexing later')
    parser.add_argument('--playlist', action='store_true', help='Expand all entries in supplied playlists')
    args = parser.parse_args()
    if not args.file and not args.url:
        parser.error('Supply urls.txt and/or --url')
    try:
        config = load_config(args.config, args.model)
        config.prepare()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                            handlers=[logging.StreamHandler(), logging.FileHandler(
                                config.path('logs') / 'ingest.log', encoding='utf-8')])
        logging.getLogger('httpx').setLevel(logging.WARNING)
        urls = list(args.url)
        if args.file:
            urls += [line.strip() for line in Path(args.file).read_text(encoding='utf-8-sig').splitlines()
                     if line.strip() and not line.lstrip().startswith('#')]
        if not urls:
            parser.error('The input URL list is empty')
        with database_lock(config.path('db') / 'pipeline.lock'):
            pipeline = Pipeline(config)
            try:
                expanded, seen, errors = [], set(), 0
                for url in urls:
                    try:
                        for item in expand_url(url, config, args.playlist):
                            key = video_id(item)
                            if key not in seen:
                                expanded.append(item)
                                seen.add(key)
                    except Exception as exc:
                        pipeline.store.fail(None, url, exc)
                        logging.exception('Failed to expand %s', url)
                        errors += 1
                errors += pipeline.run(expanded, args.force, args.skip_embeddings)
                logging.info('Finished: %d unique videos, %d failures.', len(expanded), errors)
                return 1 if errors else 0
            finally:
                pipeline.close()
    except KeyboardInterrupt:
        logging.warning('Interrupted. Run the same command to resume.')
        return 130
    except Exception:
        logging.exception('Ingestion failed')
        return 1


if __name__ == '__main__':
    sys.exit(main())
