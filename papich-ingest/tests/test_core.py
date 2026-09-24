from pathlib import Path
import pytest
from src.chunker import chunk_segments
from src.cleaner import clean_segments
from src.config import Config
from src.dedup import annotate, similarity
from src.metadata_store import MetadataStore
from src.models import Video
from src.urls import video_id, canonical_url, timestamp_url, playlist_id
from src.utils import write_json

ID = 'dQw4w9WgXcQ'


@pytest.mark.parametrize('url', [ID, f'https://youtu.be/{ID}?t=20',
    f'https://www.youtube.com/watch?v={ID}&list=PL123',
    f'https://youtube.com/shorts/{ID}', f'https://m.youtube.com/live/{ID}',
    f'https://www.youtube-nocookie.com/embed/{ID}'])
def test_urls(url):
    assert video_id(url) == ID
    assert canonical_url(url) == f'https://www.youtube.com/watch?v={ID}'


@pytest.mark.parametrize('url', ['https://evil.youtube.com/watch?v=' + ID,
    'https://youtube.com.evil.org/watch?v=' + ID, 'https://youtube.com/playlist?list=123',
    'https://youtube.com/watch?v=../../../file', 'garbage'])
def test_invalid_urls(url):
    assert video_id(url) is None


def test_timestamp():
    assert timestamp_url(ID, 534.9).endswith('&t=534s')
    assert timestamp_url(ID, -1).endswith('&t=0s')
    assert playlist_id('https://www.youtube.com/playlist?list=PL123') == 'PL123'


def test_chunk_boundaries_overlap_and_coverage():
    config = Config(chunk_max_tokens=32, chunk_overlap_tokens=9, chunk_target_words=450)
    segments = [{'start': i * 2, 'end': i * 2 + 2,
                 'text': f'word{i} one two three', 'speaker': None} for i in range(24)]
    video = Video(ID, 'Title', 'Channel', canonical_url(ID), canonical_url(ID)).to_dict()
    chunks = chunk_segments(segments, video, config, lambda t: len(t.split()) + 2)
    assert len(chunks) > 1
    assert all(c['token_count'] <= 32 for c in chunks)
    assert chunks[1]['start'] < chunks[0]['end']
    assert chunks[-1]['end'] == 48
    assert all(f'word{i}' in ' '.join(c['text'] for c in chunks) for i in range(24))
    assert len({c['id'] for c in chunks}) == len(chunks)


def test_oversized_segment_and_empty():
    config = Config(chunk_max_tokens=32, chunk_overlap_tokens=4)
    video = Video(ID, '', '', canonical_url(ID), '').to_dict()
    count = lambda t: len(t.split()) + 2
    chunks = chunk_segments([{'start': 1, 'end': 200, 'text': 'речь ' * 100}], video, config, count)
    assert len(chunks) == 4
    assert all(c['start'] == 1 and c['end'] == 200 for c in chunks)
    assert chunk_segments([], video, config, count) == []


def test_large_next_segment_does_not_emit_overlap_only_chunk():
    config = Config(chunk_max_tokens=32, chunk_overlap_tokens=16)
    video = Video(ID, '', '', canonical_url(ID), '').to_dict()
    segments = [{'start': i, 'end': i + 1, 'text': word * count}
                for i, (word, count) in enumerate([('a ', 8), ('b ', 8), ('c ', 30)])]
    chunks = chunk_segments(segments, video, config, lambda t: len(t.split()) + 2)
    assert len(chunks) == 2
    assert chunks[1]['text'].startswith('c ')


def test_cleaning_preserves_real_repetition():
    raw = [{'start': 0, 'end': 1, 'text': '  да   да  '},
           {'start': 2, 'end': 3, 'text': 'да да'},
           {'start': 4, 'end': 5, 'text': ' '},
           {'start': 2, 'end': 3, 'text': 'да да'}]
    result = clean_segments(raw)
    assert [r['text'] for r in result] == ['да да', 'да да']
    assert raw[0]['text'] == '  да   да  '


def test_sqlite_resume_and_artifact_corruption(tmp_path):
    store = MetadataStore(tmp_path / 'db.sqlite')
    video = Video(ID, 'Title', 'Channel', canonical_url(ID), '')
    store.discover(video)
    store.advance(ID, 'downloaded')
    store.fail(ID, video.url, 'ASR error')
    assert store.get(ID)['status'] == 'failed'
    assert store.get(ID)['last_success'] == 'downloaded'
    store.close()
    store = MetadataStore(tmp_path / 'db.sqlite')
    store.advance(ID, 'transcribed')
    assert store.get(ID)['error'] is None
    with pytest.raises(ValueError):
        store.advance(ID, 'downloaded')
    path = tmp_path / 'raw.json'
    write_json(path, {'segments': []})
    store.artifact(ID, 'raw', path)
    assert store.valid_artifact(ID, 'raw')
    path.write_text('{}')
    assert not store.valid_artifact(ID, 'raw')
    store.rewind(ID)
    assert store.get(ID)['last_success'] == 'discovered'
    store.close()


def test_dedup_annotations_no_deletion():
    text = 'я думаю что работа и деньги важны но отдых тоже нужен'
    assert similarity(text, text.upper() + '!') == 1
    assert similarity(text, 'совершенно другое предложение') == 0
    chunks = [{'id': 'a', 'text': text}, {'id': 'b', 'text': 'совершенно иная речь'}]
    annotate(chunks, [{'id': 'older', 'text': text}], .92)
    assert len(chunks) == 2
    assert chunks[0]['duplicate_of'] == 'older'
    assert chunks[1]['duplicate_of'] is None
    assert similarity('да', 'нет') == 0


def test_config_profile_and_invalid_overlap(tmp_path):
    with pytest.raises(ValueError):
        Config(chunk_overlap_tokens=480)
    store = MetadataStore(tmp_path / 'metadata.sqlite')
    store.ensure_index_profile('one')
    store.ensure_index_profile('one')
    with pytest.raises(ValueError):
        store.ensure_index_profile('two')
    store.close()
