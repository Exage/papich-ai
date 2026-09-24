"""Imported only in an isolated worker; exiting releases MLX model caches."""
from pathlib import Path
import logging
import math
import platform
from .audio import duration, extract_block
from .models import validate_segments
from .utils import read_json, write_json, digest

log = logging.getLogger(__name__)
MODELS = {name: f'mlx-community/whisper-{name}-mlx'
          for name in ('tiny', 'base', 'small', 'medium', 'large-v3')}
MODELS['turbo'] = 'mlx-community/whisper-large-v3-turbo'


def transcribe(audio_path, output_path, video, config):
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('MLX Whisper requires native Apple Silicon Python (arm64, macOS).')
    import mlx_whisper
    audio_path, output_path = Path(audio_path), Path(output_path)
    total = duration(audio_path)
    parts = output_path.parent / (video['id'] + '.parts')
    parts.mkdir(exist_ok=True)
    # Includes the actual audio digest: checkpoints never cross incompatible audio/config.
    signature = config.fingerprint('asr') + ':' + digest(audio_path)
    segments, checkpoints = [], []
    count = math.ceil(total / config.audio_block_seconds)
    for index in range(count):
        start = index * config.audio_block_seconds
        length = min(config.audio_block_seconds, total - start)
        checkpoint = parts / f'{index:06d}.json'
        saved = None
        try:
            candidate = read_json(checkpoint)
            if candidate['signature'] == signature and candidate['sha256'] == digest(parts / f'{index:06d}.raw.json'):
                saved = read_json(parts / f'{index:06d}.raw.json')
                validate_segments(saved['segments'])
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if saved is None:
            log.info('ASR block %d/%d (%.0f–%.0f sec)', index + 1, count, start, start + length)
            block = parts / 'current.wav'
            extract_block(audio_path, block, start, length)
            result = mlx_whisper.transcribe(
                str(block), path_or_hf_repo=MODELS.get(config.whisper_model, config.whisper_model),
                language=config.language or None, task='transcribe',
                verbose=None, condition_on_previous_text=False,
                temperature=0.0, word_timestamps=False)
            validate_segments(result['segments'])
            # Raw ASR output retains confidence/no_speech/etc and local block timestamps.
            saved = {'offset': start, 'duration': length, 'segments': result['segments'],
                     'text': result.get('text', ''), 'language': result.get('language')}
            raw_part = parts / f'{index:06d}.raw.json'
            write_json(raw_part, saved)
            write_json(checkpoint, {'signature': signature, 'sha256': digest(raw_part)})
            block.unlink(missing_ok=True)
        checkpoints.append({'offset': start, 'duration': length, 'file': f'{index:06d}.raw.json'})
        for segment in saved['segments']:
            item = dict(segment)
            item['start'] = min(total, start + min(length, segment['start']))
            item['end'] = min(total, start + min(length, segment['end']))
            segments.append(item)
    write_json(output_path, {'video': video, 'model': config.whisper_model,
                            'language': config.language, 'audio_duration': total,
                            'blocks': checkpoints, 'segments': segments})
