import json
import sys
from types import SimpleNamespace
import wave
import pytest
from src.config import Config
from src.transcriber import transcribe
from src.utils import read_json


def test_asr_block_resume_and_corrupt_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr('platform.system', lambda: 'Darwin')
    monkeypatch.setattr('platform.machine', lambda: 'arm64')
    calls = []
    should_fail = [True]
    def fake_transcribe(path, **kwargs):
        calls.append(path)
        if len(calls) == 2 and should_fail[0]:
            raise RuntimeError('power lost')
        return {'text': 'тест', 'language': 'ru', 'segments': [
            {'start': 0., 'end': 1., 'text': 'тест', 'avg_logprob': -0.1}]}
    monkeypatch.setitem(sys.modules, 'mlx_whisper', SimpleNamespace(transcribe=fake_transcribe))
    audio = tmp_path / 'audio.wav'
    with wave.open(str(audio), 'wb') as out:
        out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        out.writeframes(b'\0\0' * 16000 * 61)
    output = tmp_path / 'video.raw.json'
    video = {'id': 'abcdefghijk'}
    cfg = Config(audio_block_seconds=30)
    with pytest.raises(RuntimeError):
        transcribe(audio, output, video, cfg)
    assert not output.exists()
    should_fail[0] = False
    transcribe(audio, output, video, cfg)
    assert len(calls) == 4  # first completed block is reused
    assert [s['start'] for s in read_json(output)['segments']] == [0., 30., 60.]
    assert read_json(output)['segments'][0]['avg_logprob'] == -0.1
    part = tmp_path / 'abcdefghijk.parts' / '000001.raw.json'
    part.write_text('{}')
    transcribe(audio, output, video, cfg)
    assert len(calls) == 5
    cfg.language = 'en'
    transcribe(audio, output, video, cfg)
    assert len(calls) == 8


def test_diarization_labels_by_max_overlap():
    from src.diarization import assign_speakers
    segments = [{'start': 0, 'end': 3, 'text': 'один'},
                {'start': 4, 'end': 5, 'text': 'два'},
                {'start': 8, 'end': 9, 'text': 'неизвестно'}]
    result = assign_speakers(segments, [(0, 2, 'speaker_0'), (2, 6, 'speaker_1')])
    assert [r['speaker'] for r in result] == ['speaker_0', 'speaker_1', None]
