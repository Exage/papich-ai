"""Optional local pyannote adapter; speaker labels are local to one recording."""
import os
from .utils import read_json, write_json


def assign_speakers(segments, turns):
    result = []
    for segment in segments:
        overlaps = {}
        for start, end, speaker in turns:
            overlap = max(0, min(end, segment['end']) - max(start, segment['start']))
            overlaps[speaker] = overlaps.get(speaker, 0) + overlap
        winner = max(overlaps, key=overlaps.get) if overlaps else None
        speaker = winner if winner and overlaps[winner] > 0 else None
        result.append(dict(segment, speaker=speaker))
    return result


def diarize(audio_path, raw_path, output_path, config):
    try:
        from pyannote.audio import Pipeline
        import soundfile as sf
        import torch
    except ImportError as exc:
        raise RuntimeError('Install requirements-diarization.txt and follow README HF setup') from exc
    # Only this free local model (or a local directory); no hosted precision API.
    if config.diarization_model != 'pyannote/speaker-diarization-community-1':
        from pathlib import Path
        if not Path(config.diarization_model).is_dir():
            raise ValueError('Use community-1 or an existing local model directory')
    pipeline = Pipeline.from_pretrained(config.diarization_model, token=os.getenv('HF_TOKEN'))
    if pipeline is None:
        raise RuntimeError('Accept community-1 model terms and set HF_TOKEN (read access)')
    # Avoid torchcodec/FFmpeg ABI coupling by supplying a waveform directly.
    waveform, rate = sf.read(str(audio_path), dtype='float32', always_2d=True)
    output = pipeline({'waveform': torch.from_numpy(waveform.T.copy()), 'sample_rate': rate})
    annotation = output.exclusive_speaker_diarization
    labels, turns = {}, []
    for turn, _, label in annotation.itertracks(yield_label=True):
        speaker = labels.setdefault(label, f'speaker_{len(labels)}')
        turns.append((turn.start, turn.end, speaker))
    raw = read_json(raw_path)
    write_json(output_path, {'segments': assign_speakers(raw['segments'], turns),
                             'turns': turns, 'model': config.diarization_model})
