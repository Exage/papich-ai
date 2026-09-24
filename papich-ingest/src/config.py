from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json


@dataclass
class Config:
    data_dir: str = 'data'
    audio_dir: str = 'audio'
    transcripts_dir: str = 'transcripts'
    chunks_dir: str = 'chunks'
    db_dir: str = 'db'
    logs_dir: str = 'logs'
    whisper_model: str = 'small'
    language: str = 'ru'
    audio_block_seconds: int = 600
    embedding_model: str = 'intfloat/multilingual-e5-small'
    embedding_device: str = 'cpu'
    embedding_batch_size: int = 8
    passage_prefix: str = 'passage: '
    query_prefix: str = 'query: '
    chunk_target_words: int = 450
    chunk_max_tokens: int = 480
    chunk_overlap_tokens: int = 64
    top_k: int = 5
    dedup_threshold: float = 0.92
    delete_audio_after_processing: bool = True
    diarization_enabled: bool = False
    diarization_model: str = 'pyannote/speaker-diarization-community-1'
    cookies_file: str | None = None

    def __post_init__(self):
        if not 0 <= self.chunk_overlap_tokens < self.chunk_max_tokens:
            raise ValueError('Require 0 <= chunk_overlap_tokens < chunk_max_tokens')
        if not 32 <= self.chunk_max_tokens <= 510:
            raise ValueError('chunk_max_tokens must be 32..510 for this E5 pipeline')
        if min(self.chunk_target_words, self.embedding_batch_size, self.top_k) < 1:
            raise ValueError('Word target, batch size and top_k must be positive')
        if self.audio_block_seconds < 30 or not 0 <= self.dedup_threshold <= 1:
            raise ValueError('Invalid audio_block_seconds or dedup_threshold')

    def path(self, name):
        return Path(self.data_dir) / getattr(self, name + '_dir')

    def prepare(self):
        for name in ('audio', 'transcripts', 'chunks', 'db', 'logs'):
            self.path(name).mkdir(parents=True, exist_ok=True)

    def fingerprint(self, stage):
        keys = {
            'asr': ['whisper_model', 'language', 'audio_block_seconds'],
            'chunks': ['embedding_model', 'passage_prefix', 'chunk_target_words',
                       'chunk_max_tokens', 'chunk_overlap_tokens', 'dedup_threshold',
                       'diarization_enabled', 'diarization_model'],
            'index': ['embedding_model', 'passage_prefix', 'query_prefix'],
        }[stage]
        return hashlib.sha256(json.dumps({k: getattr(self, k) for k in keys},
                                        sort_keys=True).encode()).hexdigest()


def load_config(path='config.yaml', model=None):
    import yaml
    file = Path(path).resolve()
    values = yaml.safe_load(file.read_text()) if file.exists() else {}
    if values is not None and not isinstance(values, dict):
        raise ValueError('Config must be a YAML mapping')
    values = values or {}
    if model:
        values['whisper_model'] = model
    config = Config(**values)
    config.data_dir = str((file.parent / config.data_dir).resolve())
    if config.cookies_file:
        config.cookies_file = str((file.parent / config.cookies_file).resolve())
    return config
