"""Replace this adapter to use another TTS engine without changing the API."""
import os
import gc
import json
import random
import subprocess
from pathlib import Path

MODEL_ID = os.environ.get("VOICE_LAB_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit")
ENGINES = {
    "qwen": {"name": "Qwen3-TTS 1.7B", "model": MODEL_ID},
    "chatterbox": {"name": "Chatterbox Multilingual V3", "model": "mlx-community/chatterbox-multilingual-v3"},
    "xtts": {"name": "XTTS v2", "model": "tts_models/multilingual/multi-dataset/xtts_v2"},
}


class EngineError(RuntimeError):
    pass


CHATTERBOX_GENERATE_ARGS = {
    "exaggeration",
    "cfg_weight",
    "temperature",
    "repetition_penalty",
    "min_p",
    "top_p",
    "max_new_tokens",
    "lang_code",
    "verbose",
}


class VoiceEngine:
    def __init__(self, model_cache: Path):
        os.environ.setdefault("HF_HOME", str(model_cache))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        self.model = None
        self.state = "not_loaded"
        self.error = None
        self.active = None
        self.model_cache = model_cache

    def unload(self):
        self.model = None
        gc.collect()
        if self.active in ("qwen", "chatterbox"):
            import mlx.core as mx
            mx.clear_cache()
        self.active = None
        self.state = "not_loaded"

    def _xtts(self, request):
        python = Path(__file__).parent / ".venv-xtts/bin/python"
        if not python.exists():
            raise EngineError("XTTS не установлен. Выполните установку requirements-xtts.txt из README.")
        env = dict(os.environ, TTS_HOME=str(self.model_cache / "coqui"),
                   MPLCONFIGDIR=str(self.model_cache / "matplotlib"),
                   TORCH_HOME=str(self.model_cache / "torch"), PYTORCH_ENABLE_MPS_FALLBACK="1")
        log = self.model_cache.parent / "xtts.log"
        try:
            with log.open("w") as stream:
                result = subprocess.run([str(python), str(Path(__file__).with_name("xtts_runner.py"))],
                    input=json.dumps(request), text=True, stdout=stream, stderr=stream, env=env, timeout=900)
        except subprocess.TimeoutExpired:
            raise EngineError("XTTS превысил 15 минут. Подробности: data/xtts.log.")
        if result.returncode:
            raise EngineError("Ошибка XTTS. Подробности: data/xtts.log. Можно переключиться на другой движок.")

    def load(self, engine_id="qwen", license_accepted=False):
        if engine_id not in ENGINES:
            raise EngineError("Неизвестный движок.")
        if engine_id == "xtts" and not license_accepted:
            raise EngineError("Для XTTS подтвердите условия лицензии в интерфейсе.")
        if self.active == engine_id and self.state == "ready":
            return
        self.unload()
        self.active = engine_id
        self.state, self.error = "loading", None
        try:
            if engine_id == "xtts":
                self._xtts({"license_accepted": license_accepted})
            else:
                from mlx_audio.tts.utils import load_model
                self.model = load_model(ENGINES[engine_id]["model"])
            self.state = "ready"
        except Exception as exc:
            self.state, self.error = "error", str(exc)
            raise

    def synthesize(self, text: str, reference: Path, output: Path, transcript: str = "",
                   engine_id="qwen", chatterbox_settings=None, license_accepted=False):
        if engine_id == "xtts":
            if not license_accepted:
                raise EngineError("Для XTTS подтвердите условия лицензии в интерфейсе.")
            self.unload()
            self.active, self.state = engine_id, "loading"
            try:
                self._xtts({"license_accepted": True, "text": text, "reference": str(reference), "output": str(output)})
                self.state, self.error = "ready", None
            except Exception as exc:
                self.state, self.error = "error", str(exc)
                raise
            import soundfile as sf
            info = sf.info(str(output))
            return round(info.duration, 2), None
        self.load(engine_id)
        import numpy as np
        import soundfile as sf
        import mlx.core as mx

        chunks = []
        rate = None
        used_seed = None
        if engine_id == "qwen":
            params = {"ref_text": transcript or None, "lang_code": "russian"}
            generate_kwargs = {"max_tokens": 1500, "verbose": False, "ref_audio": str(reference), **params}
        elif engine_id == "chatterbox":
            settings = chatterbox_settings or {}
            seed = settings.get("seed")
            if seed is None:
                seed = random.randint(0, 4294967295)
            mx.random.seed(seed)
            used_seed = seed
            params = {k: v for k, v in settings.items() if k in CHATTERBOX_GENERATE_ARGS}
            if "max_new_tokens" not in params:
                params["max_new_tokens"] = 1500
            generate_kwargs = {"ref_audio": str(reference), **params}
        else:
            raise EngineError(f"Неизвестный движок: {engine_id}")

        for result in self.model.generate(text=text, **generate_kwargs):
            chunks.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
            rate = result.sample_rate
        if not chunks:
            raise RuntimeError("Модель не вернула аудио.")
        audio = np.concatenate(chunks)
        if not len(audio) or not np.isfinite(audio).all():
            raise RuntimeError("Модель вернула некорректное аудио.")
        sf.write(str(output), audio, rate, subtype="PCM_16")
        return round(len(audio) / rate, 2), used_seed
