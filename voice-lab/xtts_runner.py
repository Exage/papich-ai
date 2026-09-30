"""Isolated Coqui environment; JSON on stdin, progress/errors in the server log."""
import json
import sys
import os

request = json.load(sys.stdin)
if not request.get("license_accepted"):
    raise RuntimeError("Сначала ознакомьтесь с лицензией XTTS и подтвердите её в интерфейсе.")
os.environ["COQUI_TOS_AGREED"] = "1"
import torch
from TTS.api import TTS

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
if request.get("text"):
    model.tts_to_file(text=request["text"], speaker_wav=request["reference"],
                      language="ru", file_path=request["output"], split_sentences=True)
