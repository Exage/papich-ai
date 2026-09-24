import wave
import pytest
from src.config import Config
from src.models import Video
from src.pipeline import Pipeline
from src.utils import write_json, write_jsonl, read_jsonl
from src.vector_store import VectorStore

ID = 'dQw4w9WgXcQ'
URL = f'https://www.youtube.com/watch?v={ID}'


@pytest.fixture
def harness(tmp_path, monkeypatch):
    cfg = Config(data_dir=str(tmp_path), delete_audio_after_processing=False)
    calls = []
    monkeypatch.setattr('src.downloader.metadata', lambda url, config: Video(ID, 'Test', '', URL, url))

    def download(video, directory, config):
        calls.append('download')
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'source.wav'
        with wave.open(str(path), 'wb') as out:
            out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            out.writeframes(b'\0\0' * 16000)
        return path

    monkeypatch.setattr('src.downloader.download', download)
    monkeypatch.setattr('src.audio.normalize', lambda source, out: out.write_bytes(source.read_bytes()))

    def worker(stage, config, **job):
        calls.append(stage)
        if stage == 'transcribe':
            write_json(job['raw'], {'video': job['video'], 'segments': [
                {'start': 0, 'end': 1, 'text': 'деньги и работа'}]})
        elif stage == 'chunk':
            write_json(job['clean'], {'segments': []})
            write_jsonl(job['chunks'], [{'id': ID + '_000000', 'video_id': ID,
                                       'text': 'деньги и работа'}])
        elif stage == 'embed':
            store = VectorStore(config.path('db') / 'vectors', 3)
            store.delete_video(ID)
            store.upsert(read_jsonl(job['chunks']), [[1., 0., 0.]])
            store.close()

    pipeline = Pipeline(cfg, worker)
    yield pipeline, calls, worker
    pipeline.close()


def test_skip_then_embed_no_retranscription_and_force(harness):
    pipeline, calls, _ = harness
    assert pipeline.run([URL], skip_embeddings=True) == 0
    assert pipeline.store.get(ID)['status'] == 'chunked'
    assert 'embed' not in calls
    assert pipeline.run([URL]) == 0
    assert pipeline.store.get(ID)['status'] == 'completed'
    assert calls.count('transcribe') == 1
    assert calls.count('download') == 1
    previous = list(calls)
    assert pipeline.run([URL]) == 0
    assert calls == previous
    assert pipeline.run([URL], force=True) == 0
    assert calls.count('transcribe') == 2
    assert pipeline.vector_count(ID) == 1


def test_failure_resumes_at_chunks(harness):
    pipeline, calls, worker = harness
    def fail(stage, config, **job):
        if stage == 'embed':
            raise RuntimeError('simulated crash')
        worker(stage, config, **job)
    pipeline.worker = fail
    assert pipeline.run([URL]) == 1
    assert pipeline.store.get(ID)['status'] == 'failed'
    assert pipeline.store.get(ID)['last_success'] == 'chunked'
    pipeline.worker = worker
    assert pipeline.run([URL]) == 0
    assert calls.count('transcribe') == 1
    assert calls.count('chunk') == 1


def test_corrupt_chunk_rebuilt_from_raw(harness):
    pipeline, calls, _ = harness
    pipeline.run([URL])
    (pipeline.config.path('chunks') / f'{ID}.jsonl').write_text('{bad json')
    assert pipeline.run([URL]) == 0
    assert calls.count('chunk') == 2
    assert calls.count('transcribe') == 1
    assert pipeline.vector_count(ID) == 1


def test_deleted_vectors_reindexed_without_audio(harness):
    pipeline, calls, _ = harness
    pipeline.run([URL])
    pipeline.clear_vectors(ID)
    assert pipeline.run([URL]) == 0
    assert calls.count('transcribe') == 1
    assert calls.count('embed') == 2


def test_failed_video_does_not_stop_next(harness, monkeypatch):
    pipeline, _, _ = harness
    original = pipeline.process
    failed_id = 'abcdefghijk'
    def process(url, *args):
        if failed_id in url:
            raise RuntimeError('Video unavailable')
        return original(url, *args)
    monkeypatch.setattr(pipeline, 'process', process)
    assert pipeline.run([f'https://youtu.be/{failed_id}', URL]) == 1
    assert pipeline.store.get(failed_id)['status'] == 'failed'
    assert pipeline.store.get(ID)['status'] == 'completed'


def test_vector_search_metadata_and_idempotency(tmp_path):
    store = VectorStore(tmp_path / 'vectors', 3)
    chunk = {'id': 'a', 'video_id': ID, 'text': 'деньги', 'timestamp_url': URL + '&t=4s'}
    store.upsert([chunk], [[1., 0., 0.]])
    store.upsert([chunk], [[1., 0., 0.]])
    assert store.count_video(ID) == 1
    hit = store.search([1., 0., 0.], 1, [ID])[0]
    assert hit.score == pytest.approx(1.)
    assert hit.payload['timestamp_url'].endswith('&t=4s')
    assert store.search([1., 0., 0.], 1, []) == []
    store.close()
