from pathlib import Path
import json
import logging
import shutil
from . import audio, downloader
from .dedup import annotate
from .metadata_store import MetadataStore
from .models import Video
from .urls import video_id, canonical_url
from .utils import read_jsonl, write_jsonl
from .worker import run_worker

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config, worker=run_worker):
        self.config, self.worker = config, worker
        config.prepare()
        self.store = MetadataStore(config.path('db') / 'metadata.sqlite')

    def vector_count(self, key):
        from .vector_store import VectorStore
        store = VectorStore(self.config.path('db') / 'vectors')
        try:
            return store.count_video(key)
        finally:
            store.close()

    def clear_vectors(self, key):
        from .vector_store import VectorStore
        store = VectorStore(self.config.path('db') / 'vectors')
        try:
            store.delete_video(key)
        finally:
            store.close()

    def ensure_audio(self, video):
        config, store = self.config, self.store
        directory = config.path('audio') / video.id
        wav = directory / 'normalized.wav'
        if store.valid_artifact(video.id, 'audio'):
            try:
                audio.duration(wav)
                return wav
            except (OSError, ValueError, EOFError):
                pass
        source = None
        if store.valid_artifact(video.id, 'source'):
            candidates = [p for p in directory.glob('source.*') if p.suffix not in ('.part', '.ytdl')]
            source = candidates[0] if len(candidates) == 1 else None
        if source is None:
            # A corrupt source must not be reused by yt-dlp's existing-file check.
            if directory.exists():
                shutil.rmtree(directory)
            source = downloader.download(video, directory, config)
            store.artifact(video.id, 'source', source)
        audio.normalize(source, wav)
        store.artifact(video.id, 'audio', wav)
        return wav

    def process(self, url, prefix='', force=False, skip_embeddings=False):
        config, store = self.config, self.store
        key = video_id(url)
        if not key:
            raise ValueError('Invalid video URL')
        row = store.get(key)
        if row and row['asr_key'] and not force:
            video = Video(**json.loads(row['metadata']))
        else:
            video = downloader.metadata(url, config)
            store.discover(video)
            row = store.get(key)
        asr_key, chunks_key = config.fingerprint('asr'), config.fingerprint('chunks')
        if not skip_embeddings:
            store.ensure_index_profile(config.fingerprint('index'))
        raw = config.path('transcripts') / f'{key}.raw.json'
        clean = config.path('transcripts') / f'{key}.json'
        diarized = config.path('transcripts') / f'{key}.speakers.json'
        chunks_path = config.path('chunks') / f'{key}.jsonl'
        raw_valid = not force and row['asr_key'] == asr_key and store.valid_artifact(key, 'raw')
        chunks_valid = (raw_valid and row['chunks_key'] == chunks_key
                        and store.valid_artifact(key, 'clean') and store.valid_artifact(key, 'chunks'))
        if chunks_valid:
            chunks = read_jsonl(chunks_path)
            if row['status'] == 'completed' and row['indexed']:
                if self.vector_count(key) == len(chunks):
                    log.info('%s Skipped: %s', prefix, video.title)
                    return 'skipped'
        if force:
            store.invalidate(key, ['audio', 'source'])
            parts = config.path('transcripts') / f'{key}.parts'
            if parts.exists():
                shutil.rmtree(parts)
        if not chunks_valid:
            self.clear_vectors(key)
            store.replace_chunks(key, [])
            store.invalidate(key, ['clean', 'chunks', 'diarized'])
            if not raw_valid:
                store.invalidate(key, ['raw'])
        store.rewind(key, 'chunked' if chunks_valid else ('transcribed' if raw_valid else 'discovered'))
        store.set_keys(key, asr_key, chunks_key)
        job = {'video': video.to_dict(), 'raw': str(raw)}
        if not raw_valid:
            log.info('%s Downloading: %s', prefix, video.title)
            wav = self.ensure_audio(video)
            store.advance(key, 'downloaded')
            log.info('%s Transcribing: %s', prefix, video.title)
            self.worker('transcribe', config, **job, audio=str(wav))
            store.artifact(key, 'raw', raw)
            store.advance(key, 'transcribed')
        if not chunks_valid:
            if config.diarization_enabled:
                log.info('%s Diarizing: %s', prefix, video.title)
                wav = self.ensure_audio(video)
                self.worker('diarize', config, **job, audio=str(wav), diarized=str(diarized))
                store.artifact(key, 'diarized', diarized)
            log.info('%s Chunking: %s', prefix, video.title)
            self.worker('chunk', config, **job, clean=str(clean), chunks=str(chunks_path),
                        diarized=str(diarized) if config.diarization_enabled else None)
            chunks = annotate(read_jsonl(chunks_path), store.candidates(key), config.dedup_threshold)
            write_jsonl(chunks_path, chunks)
            store.replace_chunks(key, chunks)
            store.artifact(key, 'clean', clean)
            store.artifact(key, 'chunks', chunks_path)
            store.advance(key, 'chunked')
        if skip_embeddings:
            log.info('%s Saved transcript/chunks; embeddings pending.', prefix)
            return 'chunked'
        if chunks:
            log.info('%s Embedding: %s (%d chunks)', prefix, video.title, len(chunks))
            self.worker('embed', config, video=video.to_dict(), chunks=str(chunks_path))
        else:
            self.clear_vectors(key)
            log.warning('%s No speech segments: saved empty transcript; no vectors.', prefix)
        if self.vector_count(key) != len(chunks):
            raise RuntimeError('Vector count mismatch after embedding; retry this video')
        store.advance(key, 'embedded', indexed=True)
        store.advance(key, 'completed', indexed=True)
        if config.delete_audio_after_processing:
            directory = config.path('audio') / key
            if directory.exists():
                shutil.rmtree(directory)
        log.info('%s Saved: %s', prefix, video.title)
        return 'completed'

    def run(self, urls, force=False, skip_embeddings=False):
        failures = 0
        for index, url in enumerate(urls, 1):
            prefix = f'[{index}/{len(urls)}]'
            try:
                self.process(url, prefix, force, skip_embeddings)
            except Exception as exc:
                key = video_id(url)
                if key and not self.store.get(key):
                    self.store.discover(Video(key, key, '', canonical_url(key), url))
                self.store.fail(key, url, exc)
                log.exception('%s Failed: %s', prefix, url)
                failures += 1
        return failures

    def close(self):
        self.store.close()
