import io
import wave
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import app as lab


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, 'XTTS_CONSENT', tmp_path / 'xtts-license.json')
    for name in ("VOICES", "OUTPUTS"):
        directory = tmp_path / name.lower()
        directory.mkdir()
        monkeypatch.setattr(lab, name, directory)
    with TestClient(lab.app) as test_client:
        yield test_client


def wav_bytes(seconds=4):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\0\0" * int(24000 * seconds))
    return buffer.getvalue()


def test_upload_crop_and_retrieve(client):
    response = client.post("/api/voices", files={"file": ("sample.wav", wav_bytes())},
                           data={"name": "Тест", "start": "0.5", "seconds": "3"})
    assert response.status_code == 201, response.text
    voice = response.json()
    assert voice["duration"] == 3
    assert client.get(voice["audioUrl"]).content[:4] == b"RIFF"
    assert client.get("/api/voices").json()[0]["name"] == "Тест"


def test_reject_invalid_and_short_audio_without_orphans(client):
    for content in (b"invalid", wav_bytes(1)):
        response = client.post("/api/voices", files={"file": ("sample.wav", content)})
        assert response.status_code == 422
    assert list(lab.VOICES.iterdir()) == []


def test_validation_and_origin_protection(client):
    assert client.post("/api/chat", json={"message": "  "}).status_code == 422
    assert client.post("/api/speech", json={"text": "hello", "voice": "../secret"}).status_code == 404
    assert client.post("/api/tts/prepare", headers={"origin": "https://example.com"}).status_code == 403


def test_chat_disallows_nonlocal_model(client, monkeypatch):
    monkeypatch.setattr(lab, "status", AsyncMock(return_value={"ollama": {"error": None, "models": ["gemma4:e2b"]}}))
    response = client.post("/api/chat", json={"message": "Привет", "model": "model:cloud"})
    assert response.status_code == 422


def test_failed_speech_releases_lock_and_reports_error(client, monkeypatch):
    import time
    voice = client.post("/api/voices", files={"file": ("sample.wav", wav_bytes())}).json()
    def fail(*args, **kwargs):
        raise RuntimeError("Simulated inference failure")
    monkeypatch.setattr(lab.engine, "synthesize", fail)
    result = client.post("/api/speech", json={"text": "Привет", "voice": voice["id"], "engine": "qwen"})
    assert result.status_code == 202
    for _ in range(100):
        job = client.get(f'/api/jobs/{result.json()["id"]}').json()
        if job["state"] != "running":
            break
        time.sleep(.01)
    assert job["state"] == "error"
    assert not lab.generation_lock.locked()
    assert list(lab.OUTPUTS.iterdir()) == []


def test_engine_validation_and_xtts_consent(client):
    assert client.post('/api/tts/prepare', json={'engine': 'unknown'}).status_code == 422
    assert client.post('/api/tts/prepare', json={'engine': 'xtts'}).status_code == 422
    assert client.post('/api/tts/prepare', json={'engine': 'chatterbox', 'exaggeration': 2}).status_code == 422


@pytest.mark.parametrize('engine_id', ['qwen', 'chatterbox', 'xtts'])
def test_engine_routing_and_metadata(client, monkeypatch, engine_id):
    import time
    calls = []
    def synthesize(text, reference, output, transcript, selected, chatterbox_settings, accepted):
        calls.append((selected, chatterbox_settings, accepted))
        output.write_bytes(wav_bytes())
        return 4, None
    monkeypatch.setattr(lab.engine, 'synthesize', synthesize)
    voice = client.post('/api/voices', files={'file': ('test.wav', wav_bytes())}).json()
    response = client.post('/api/speech', json={'text': 'Привет', 'voice': voice['id'],
        'engine': engine_id, 'exaggeration': .7, 'xtts_license_accepted': engine_id == 'xtts'})
    assert response.status_code == 202
    for _ in range(100):
        job = client.get('/api/jobs/' + response.json()['id']).json()
        if job['state'] != 'running':
            break
        time.sleep(.01)
    assert job['state'] == 'done'
    assert job['engine'] == engine_id
    if engine_id == 'chatterbox':
        assert calls[0][1] is not None
        assert calls[0][1]['exaggeration'] == 0.7
    else:
        assert calls[0][1] is None or calls[0][1] == {}
    assert calls[0][2] == (engine_id == 'xtts')
    assert client.get(job['audioUrl']).status_code == 200


def test_chatterbox_settings_endpoint(client):
    response = client.get('/api/tts/settings')
    assert response.status_code == 200
    data = response.json()
    assert 'defaults' in data
    assert 'bounds' in data
    assert 'languages' in data
    defaults = data['defaults']
    assert defaults['exaggeration'] == 0.5
    assert defaults['cfg_weight'] == 0.5
    assert defaults['temperature'] == 0.8
    assert defaults['repetition_penalty'] == 1.2
    assert defaults['top_p'] == 1.0
    assert defaults['min_p'] == 0.05
    assert defaults['max_new_tokens'] == 1500
    assert defaults['lang_code'] == 'ru'
    assert defaults['seed'] is None
    assert defaults['verbose'] is False
    assert 'ru' in data['languages']


def test_speech_without_new_settings_uses_defaults(client, monkeypatch):
    import time
    voice = client.post("/api/voices", files={"file": ("sample.wav", wav_bytes())}).json()
    calls = []
    def synthesize(text, reference, output, transcript, selected, chatterbox_settings, accepted):
        calls.append((selected, chatterbox_settings))
        output.write_bytes(wav_bytes())
        return 4, None
    monkeypatch.setattr(lab.engine, 'synthesize', synthesize)
    response = client.post('/api/speech', json={'text': 'Привет', 'voice': voice['id'], 'engine': 'chatterbox'})
    assert response.status_code == 202
    for _ in range(100):
        job = client.get('/api/jobs/' + response.json()['id']).json()
        if job['state'] != 'running':
            break
        time.sleep(.01)
    assert job['state'] == 'done'
    assert calls[0][0] == 'chatterbox'
    settings = calls[0][1]
    assert settings['exaggeration'] == 0.5
    assert settings['cfg_weight'] == 0.5
    assert settings['temperature'] == 0.8
    assert settings['repetition_penalty'] == 1.2
    assert settings['top_p'] == 1.0
    assert settings['min_p'] == 0.05
    assert settings['max_new_tokens'] == 1500
    assert settings['lang_code'] == 'ru'
    assert settings['seed'] is None
    assert settings['verbose'] is False


@pytest.mark.parametrize('nested, expected', [({'temperature': 0.9}, 0.8),
                                              ({'exaggeration': 0.3}, 0.3)])
def test_initial_metadata_matches_generation_settings(client, monkeypatch, nested, expected):
    import threading
    import time
    release = threading.Event()
    calls = []

    def synthesize(text, reference, output, transcript, selected, settings, accepted):
        calls.append(settings.copy())
        assert release.wait(5)
        output.write_bytes(wav_bytes())
        return 4, 42

    monkeypatch.setattr(lab.engine, 'synthesize', synthesize)
    voice = client.post('/api/voices', files={'file': ('sample.wav', wav_bytes())}).json()
    try:
        response = client.post('/api/speech', json={
            'text': 'Привет', 'voice': voice['id'], 'engine': 'chatterbox',
            'exaggeration': 0.8, 'chatterbox': nested,
        })
        assert response.status_code == 202
        initial = response.json()
        assert initial['state'] == 'running'
        assert initial['settings']['exaggeration'] == expected
    finally:
        release.set()
    for _ in range(100):
        job = client.get('/api/jobs/' + initial['id']).json()
        if job['state'] != 'running':
            break
        time.sleep(.01)
    assert job['state'] == 'done'
    assert calls[0] == initial['settings']
    assert job['settings'] == {**initial['settings'], 'seed': 42}


def test_old_exaggeration_field_still_works(client, monkeypatch):
    import time
    voice = client.post("/api/voices", files={"file": ("sample.wav", wav_bytes())}).json()
    calls = []
    def synthesize(text, reference, output, transcript, selected, chatterbox_settings, accepted):
        calls.append((selected, chatterbox_settings))
        output.write_bytes(wav_bytes())
        return 4, None
    monkeypatch.setattr(lab.engine, 'synthesize', synthesize)
    response = client.post('/api/speech', json={'text': 'Привет', 'voice': voice['id'], 'engine': 'chatterbox', 'exaggeration': 0.8})
    assert response.status_code == 202
    for _ in range(100):
        job = client.get('/api/jobs/' + response.json()['id']).json()
        if job['state'] != 'running':
            break
        time.sleep(.01)
    assert job['state'] == 'done'
    settings = calls[0][1]
    assert settings['exaggeration'] == 0.8


def test_chatterbox_exaggeration_priority_over_old_field(client, monkeypatch):
    import time
    voice = client.post("/api/voices", files={"file": ("sample.wav", wav_bytes())}).json()
    calls = []
    def synthesize(text, reference, output, transcript, selected, chatterbox_settings, accepted):
        calls.append((selected, chatterbox_settings))
        output.write_bytes(wav_bytes())
        return 4, None
    monkeypatch.setattr(lab.engine, 'synthesize', synthesize)
    response = client.post('/api/speech', json={
        'text': 'Привет', 'voice': voice['id'], 'engine': 'chatterbox',
        'exaggeration': 0.8,
        'chatterbox': {'exaggeration': 0.3, 'cfg_weight': 0.5, 'temperature': 0.8, 'repetition_penalty': 1.2, 'top_p': 1.0, 'min_p': 0.05, 'max_new_tokens': 1500, 'lang_code': 'ru', 'seed': None, 'verbose': False}
    })
    assert response.status_code == 202
    for _ in range(100):
        job = client.get('/api/jobs/' + response.json()['id']).json()
        if job['state'] != 'running':
            break
        time.sleep(.01)
    assert job['state'] == 'done'
    settings = calls[0][1]
    assert settings['exaggeration'] == 0.3


def test_all_chatterbox_settings_reach_adapter(client, monkeypatch):
    import time
    voice = client.post("/api/voices", files={"file": ("sample.wav", wav_bytes())}).json()
    calls = []
    def synthesize(text, reference, output, transcript, selected, chatterbox_settings, accepted):
        calls.append((selected, chatterbox_settings))
        output.write_bytes(wav_bytes())
        return 4, None
    monkeypatch.setattr(lab.engine, 'synthesize', synthesize)
    custom = {'exaggeration': 0.7, 'cfg_weight': 0.8, 'temperature': 0.9, 'repetition_penalty': 1.3, 'top_p': 0.9, 'min_p': 0.1, 'max_new_tokens': 2000, 'lang_code': 'en', 'seed': 12345, 'verbose': True}
    response = client.post('/api/speech', json={'text': 'Привет', 'voice': voice['id'], 'engine': 'chatterbox', 'chatterbox': custom})
    assert response.status_code == 202
    for _ in range(100):
        job = client.get('/api/jobs/' + response.json()['id']).json()
        if job['state'] != 'running':
            break
        time.sleep(.01)
    assert job['state'] == 'done'
    settings = calls[0][1]
    for key, value in custom.items():
        assert settings[key] == value, f'{key}: expected {value}, got {settings[key]}'


def test_job_metadata_contains_applied_settings(client, monkeypatch):
    import time
    voice = client.post("/api/voices", files={"file": ("sample.wav", wav_bytes())}).json()
    def synthesize(text, reference, output, transcript, selected, chatterbox_settings, accepted):
        output.write_bytes(wav_bytes())
        return 4, 42
    monkeypatch.setattr(lab.engine, 'synthesize', synthesize)
    custom = {'exaggeration': 0.6, 'cfg_weight': 0.7, 'temperature': 0.75, 'repetition_penalty': 1.15, 'top_p': 0.95, 'min_p': 0.08, 'max_new_tokens': 1800, 'lang_code': 'ru', 'seed': 42, 'verbose': False}
    response = client.post('/api/speech', json={'text': 'Привет', 'voice': voice['id'], 'engine': 'chatterbox', 'chatterbox': custom})
    assert response.status_code == 202
    job_id = response.json()['id']
    for _ in range(100):
        job = client.get('/api/jobs/' + job_id).json()
        if job['state'] != 'running':
            break
        time.sleep(.01)
    assert job['state'] == 'done'
    assert 'settings' in job
    for key, value in custom.items():
        assert job['settings'][key] == value, f'{key}: expected {value}, got {job["settings"][key]}'


def test_auto_seed_becomes_concrete_number(client, monkeypatch):
    import time
    voice = client.post("/api/voices", files={"file": ("sample.wav", wav_bytes())}).json()
    def synthesize(text, reference, output, transcript, selected, chatterbox_settings, accepted):
        output.write_bytes(wav_bytes())
        return 4, 12345
    monkeypatch.setattr(lab.engine, 'synthesize', synthesize)
    custom = {'exaggeration': 0.5, 'cfg_weight': 0.5, 'temperature': 0.8, 'repetition_penalty': 1.2, 'top_p': 1.0, 'min_p': 0.05, 'max_new_tokens': 1500, 'lang_code': 'ru', 'seed': None, 'verbose': False}
    response = client.post('/api/speech', json={'text': 'Привет', 'voice': voice['id'], 'engine': 'chatterbox', 'chatterbox': custom})
    assert response.status_code == 202
    job_id = response.json()['id']
    for _ in range(100):
        job = client.get('/api/jobs/' + job_id).json()
        if job['state'] != 'running':
            break
        time.sleep(.01)
    assert job['state'] == 'done'
    assert 'settings' in job
    assert isinstance(job['settings']['seed'], int)
    assert 0 <= job['settings']['seed'] <= 4294967295


def test_invalid_chatterbox_values_rejected(client):
    voice = client.post("/api/voices", files={"file": ("sample.wav", wav_bytes())}).json()
    for field, bad_value in [
        ('exaggeration', -0.1), ('exaggeration', 1.1),
        ('cfg_weight', -0.1), ('cfg_weight', 2.1),
        ('temperature', 0.05), ('temperature', 1.6),
        ('repetition_penalty', 0.9), ('repetition_penalty', 2.1),
        ('top_p', 0.0), ('top_p', 1.01),
        ('min_p', -0.01), ('min_p', 1.01),
        ('max_new_tokens', 99), ('max_new_tokens', 3001),
        ('seed', -1), ('seed', 4294967296),
        ('lang_code', 'xx'),
    ]:
        payload = {'text': 'Привет', 'voice': voice['id'], 'engine': 'chatterbox', 'chatterbox': {**lab.CHATTERBOX_DEFAULTS, field: bad_value}}
        response = client.post('/api/speech', json=payload)
        assert response.status_code == 422, f'field {field} value {bad_value} should be rejected'


def test_unknown_field_in_chatterbox_rejected(client):
    voice = client.post("/api/voices", files={"file": ("sample.wav", wav_bytes())}).json()
    response = client.post('/api/speech', json={
        'text': 'Привет', 'voice': voice['id'], 'engine': 'chatterbox',
        'chatterbox': {**lab.CHATTERBOX_DEFAULTS, 'unknown_field': 123}
    })
    assert response.status_code == 422


def test_chatterbox_settings_not_passed_to_qwen_or_xtts(client, monkeypatch):
    import time
    voice = client.post("/api/voices", files={"file": ("sample.wav", wav_bytes())}).json()
    for engine_id in ['qwen', 'xtts']:
        calls = []
        def synthesize(text, reference, output, transcript, selected, chatterbox_settings, accepted):
            calls.append((selected, chatterbox_settings))
            output.write_bytes(wav_bytes())
            return 4, None
        monkeypatch.setattr(lab.engine, 'synthesize', synthesize)
        payload = {'text': 'Привет', 'voice': voice['id'], 'engine': engine_id}
        if engine_id == 'xtts':
            payload['xtts_license_accepted'] = True
        response = client.post('/api/speech', json=payload)
        assert response.status_code == 202
        for _ in range(100):
            job = client.get('/api/jobs/' + response.json()['id']).json()
            if job['state'] != 'running':
                break
            time.sleep(.01)
        assert job['state'] == 'done'
        assert calls[0][1] is None or calls[0][1] == {}


def test_generation_error_releases_lock(client, monkeypatch):
    import time
    voice = client.post("/api/voices", files={"file": ("sample.wav", wav_bytes())}).json()
    def fail(*args, **kwargs):
        raise RuntimeError("Simulated inference failure")
    monkeypatch.setattr(lab.engine, 'synthesize', fail)
    response = client.post('/api/speech', json={'text': 'Привет', 'voice': voice['id'], 'engine': 'chatterbox'})
    assert response.status_code == 202
    job_id = response.json()['id']
    for _ in range(100):
        job = client.get('/api/jobs/' + job_id).json()
        if job['state'] != 'running':
            break
        time.sleep(.01)
    assert job['state'] == 'error'
    assert not lab.generation_lock.locked()
