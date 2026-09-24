"""Local video knowledge-base preparation; no chatbot dependencies."""
import os

os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYANNOTE_METRICS_ENABLED', '0')
