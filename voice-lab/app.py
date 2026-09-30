import asyncio
import json
import logging
import os
import random
import re
import shutil
import subprocess
import time
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from voice_engine import MODEL_ID, ENGINES, EngineError, VoiceEngine

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("VOICE_LAB_DATA", ROOT / "data"))
VOICES, OUTPUTS = DATA / "voices", DATA / "outputs"
for directory in (VOICES, OUTPUTS):
    directory.mkdir(parents=True, exist_ok=True)
OLLAMA = "http://127.0.0.1:11434"
MAX_UPLOAD = 30 * 1024 * 1024
XTTS_CONSENT = DATA / "xtts-license.json"
engine = VoiceEngine(DATA / "models")
worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-model")
generation_lock = asyncio.Lock()
jobs = {}
logger = logging.getLogger("voice-lab")
app = FastAPI(title="Voice Lab", version="0.1.0")

SUPPORTED_LANGUAGES = {
    "ar": "Арабский",
    "da": "Датский",
    "de": "Немецкий",
    "el": "Греческий",
    "en": "Английский",
    "es": "Испанский",
    "fi": "Финский",
    "fr": "Французский",
    "he": "Иврит",
    "hi": "Хинди",
    "it": "Итальянский",
    "ja": "Японский",
    "ko": "Корейский",
    "ms": "Малайский",
    "nl": "Нидерландский",
    "no": "Норвежский",
    "pl": "Польский",
    "pt": "Португальский",
    "ru": "Русский",
    "sv": "Шведский",
    "sw": "Суахили",
    "tr": "Турецкий",
    "zh": "Китайский",
}

DEFAULT_LANG_CODE = "ru"

CHATTERBOX_DEFAULTS = {
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8,
    "repetition_penalty": 1.2,
    "top_p": 1.0,
    "min_p": 0.05,
    "max_new_tokens": 1500,
    "lang_code": DEFAULT_LANG_CODE,
    "seed": None,
    "verbose": False,
}

CHATTERBOX_BOUNDS = {
    "exaggeration": (0.0, 1.0),
    "cfg_weight": (0.0, 2.0),
    "temperature": (0.1, 1.5),
    "repetition_penalty": (1.0, 2.0),
    "top_p": (0.01, 1.0),
    "min_p": (0.0, 1.0),
    "max_new_tokens": (100, 3000),
    "seed": (0, 4294967295),
}
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


class ChatterboxSettings(BaseModel):
    model_config = {"extra": "forbid"}

    exaggeration: float = Field(default=0.5, ge=0.0, le=1.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=2.0)
    temperature: float = Field(default=0.8, ge=0.1, le=1.5)
    repetition_penalty: float = Field(default=1.2, ge=1.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.01, le=1.0)
    min_p: float = Field(default=0.05, ge=0.0, le=1.0)
    max_new_tokens: int = Field(default=1500, ge=100, le=3000)
    lang_code: str = Field(default=DEFAULT_LANG_CODE)
    seed: Optional[int] = Field(default=None, ge=0, le=4294967295)
    verbose: bool = Field(default=False)

    @field_validator("lang_code")
    @classmethod
    def validate_lang_code(cls, value):
        if value not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language code: {value}. Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}")
        return value

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, value):
        if value is None or value == "":
            return None
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("Seed must be an integer")
        return int(value) if value is not None else None

    @model_validator(mode="after")
    def check_no_nan_inf(self):
        for field_name in ["exaggeration", "cfg_weight", "temperature", "repetition_penalty", "top_p", "min_p"]:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"{field_name} must be a valid finite number")
        return self


class SettingsResponse(BaseModel):
    defaults: ChatterboxSettings
    bounds: dict
    languages: dict


@app.middleware("http")
async def local_requests(request: Request, call_next):
    origin = request.headers.get("origin")
    if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin != str(request.base_url).rstrip("/"):
        return JSONResponse({"detail": "Запрос разрешён только со страницы Voice Lab."}, status_code=403)
    return await call_next(request)


def record_path(voice_id: str):
    if not re.fullmatch(r"[a-f0-9]{32}", voice_id):
        raise HTTPException(404, "Голос не найден.")
    path = VOICES / f"{voice_id}.json"
    if not path.exists():
        raise HTTPException(404, "Голос не найден. Загрузите запись.")
    return path


def get_voice(voice_id: str):
    return json.loads(record_path(voice_id).read_text())


@app.get("/api/status")
async def status():
    error, models = None, []
    try:
        async with httpx.AsyncClient(timeout=4, trust_env=False) as client:
            response = await client.get(f"{OLLAMA}/api/tags")
            response.raise_for_status()
            # Only downloaded local models are allowed in this local-only prototype.
            models = [m["name"] for m in response.json().get("models", [])
                      if "cloud" not in m["name"].lower() and not m.get("remote_host")]
    except (httpx.HTTPError, ValueError, KeyError):
        error = "Нет связи с Ollama. Откройте приложение Ollama."
    return {"ollama": {"models": models, "error": error},
            "tts": {"model": ENGINES.get(engine.active, ENGINES["qwen"])["model"],
                    "engine": engine.active, "engines": ENGINES, "state": engine.state, "error": engine.error,
                    "xtts_license_accepted": XTTS_CONSENT.exists()},
            "ffmpeg": bool(shutil.which("ffmpeg")), "busy": generation_lock.locked()}


@app.get("/api/tts/settings")
def tts_settings():
    defaults = ChatterboxSettings()
    bounds = {key: {"min": val[0], "max": val[1]} for key, val in CHATTERBOX_BOUNDS.items()}
    return {"defaults": defaults.model_dump(), "bounds": bounds, "languages": SUPPORTED_LANGUAGES}


@app.get("/api/voices")
def list_voices():
    records = []
    for path in VOICES.glob("*.json"):
        try:
            records.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            logger.warning("Cannot read voice metadata: %s", path.name)
    return sorted(records, key=lambda item: item["created"], reverse=True)


def convert_audio(source: Path, target: Path, start: float, seconds: float):
    if not shutil.which("ffmpeg"):
        raise HTTPException(503, "Нужен FFmpeg: brew install ffmpeg")
    try:
        result = subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-protocol_whitelist", "file,pipe",
            "-ss", str(start), "-i", str(source), "-t", str(seconds), "-vn", "-ac", "1",
            "-ar", "24000", "-c:a", "pcm_s16le", str(target),
        ], capture_output=True, timeout=40)
        if result.returncode:
            raise HTTPException(422, "Не удалось прочитать аудио. Используйте WAV, M4A, MP3 или FLAC.")
        with wave.open(str(target)) as wav:
            duration = wav.getnframes() / wav.getframerate()
            if duration < 3:
                raise HTTPException(422, "В выбранном фрагменте меньше 3 секунд. Проверьте начало записи.")
        return round(duration, 2)
    except subprocess.TimeoutExpired:
        raise HTTPException(422, "Обработка файла заняла слишком много времени.")


@app.post("/api/voices", status_code=201)
async def upload_voice(file: UploadFile = File(...), name: str = Form("Мой голос", max_length=80),
                       start: float = Form(0, ge=0, le=3600), seconds: float = Form(15, ge=3, le=120),
                       transcript: str = Form("", max_length=3000)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".aiff", ".webm"}:
        raise HTTPException(422, "Выберите аудиофайл WAV, MP3, M4A, AAC, OGG, FLAC, AIFF или WEBM.")
    voice_id = uuid.uuid4().hex
    source, target = VOICES / f"{voice_id}.upload", VOICES / f"{voice_id}.wav"
    complete = False
    try:
        size = 0
        with source.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, "Файл больше 30 МБ. Выберите более короткую запись.")
                out.write(chunk)
        duration = await asyncio.to_thread(convert_audio, source, target, start, seconds)
        record = {"id": voice_id, "name": name.strip() or "Мой голос", "duration": duration,
                  "transcript": transcript.strip(), "start": start, "created": time.time(),
                  "audioUrl": f"/api/voices/{voice_id}/audio"}
        (VOICES / f"{voice_id}.json").write_text(json.dumps(record, ensure_ascii=False))
        complete = True
        return record
    finally:
        await file.close()
        source.unlink(missing_ok=True)
        if not complete:
            target.unlink(missing_ok=True)


@app.get("/api/voices/{voice_id}/audio")
def voice_audio(voice_id: str):
    get_voice(voice_id)
    return FileResponse(VOICES / f"{voice_id}.wav", media_type="audio/wav")


class VoiceUpdate(BaseModel):
    transcript: str = Field(max_length=3000)


@app.patch("/api/voices/{voice_id}")
async def update_voice(voice_id: str, body: VoiceUpdate):
    if generation_lock.locked():
        raise HTTPException(409, "Дождитесь завершения генерации перед изменением образца.")
    path = record_path(voice_id)
    record = get_voice(voice_id)
    record["transcript"] = body.transcript.strip()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False))
    temporary.replace(path)
    return record


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=6000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    model: str = "gemma4:e2b"
    history: list[Message] = Field(default_factory=list, max_length=20)

    @field_validator("message")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Введите вопрос.")
        return value.strip()


@app.post("/api/chat")
async def chat(body: ChatRequest):
    state = await status()
    if state["ollama"]["error"]:
        raise HTTPException(503, state["ollama"]["error"])
    if body.model not in state["ollama"]["models"]:
        raise HTTPException(422, "Выберите установленную локальную модель Ollama.")
    if generation_lock.locked():
        raise HTTPException(409, "Уже выполняется запрос. Дождитесь завершения.")
    async with generation_lock:
        started = time.monotonic()
        messages = [{"role": "system", "content": "Отвечай по-русски, кратко: 1–3 предложения. "
                     "Ответ будет прочитан вслух. Используй обычный текст без Markdown."}]
        messages += [item.model_dump() for item in body.history]
        messages.append({"role": "user", "content": body.message})
        try:
            async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
                response = await client.post(f"{OLLAMA}/api/chat", json={
                    "model": body.model, "messages": messages, "stream": False,
                    "think": False, "keep_alive": 0,
                    "options": {"num_predict": 220, "num_ctx": 4096},
                })
                response.raise_for_status()
                result = response.json()["message"]["content"].strip()
                if not result:
                    raise ValueError("empty response")
        except httpx.TimeoutException:
            raise HTTPException(504, "Gemma не ответила за 3 минуты. Повторите с более коротким вопросом.")
        except (httpx.HTTPError, ValueError, KeyError):
            logger.exception("Ollama request failed")
            raise HTTPException(502, "Ошибка ответа Ollama. Проверьте, что модель открывается в её чате.")
        return {"text": result, "seconds": round(time.monotonic() - started, 2), "model": body.model}


class EngineRequest(BaseModel):
    engine: Literal["qwen", "chatterbox", "xtts"] = "qwen"
    exaggeration: float = Field(default=0.5, ge=0, le=1)
    xtts_license_accepted: bool = False
    chatterbox: Optional[ChatterboxSettings] = None

    @model_validator(mode="after")
    def validate_chatterbox_engine_match(self):
        if self.chatterbox is not None and self.engine != "chatterbox":
            raise ValueError("chatterbox settings can only be used with engine='chatterbox'")
        return self


class SpeechRequest(EngineRequest):
    text: str = Field(min_length=1, max_length=1500)
    voice: str

    @field_validator("text")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Введите текст для озвучки.")
        return value.strip()


def resolve_chatterbox_settings(body: EngineRequest):
    if body.engine != "chatterbox":
        return None
    settings = ChatterboxSettings(exaggeration=body.exaggeration).model_dump()
    if body.chatterbox is not None:
        settings.update(body.chatterbox.model_dump(exclude_unset=True))
    return settings


async def run_job(job_id: str, body: SpeechRequest | EngineRequest):
    started = time.monotonic()
    output = OUTPUTS / f"{job_id}.wav"
    try:
        loop = asyncio.get_running_loop()
        if not isinstance(body, SpeechRequest):
            await loop.run_in_executor(worker, engine.load, body.engine, body.xtts_license_accepted)
            result = {}
        else:
            voice = get_voice(body.voice)
            chatterbox_settings = resolve_chatterbox_settings(body)
            duration, used_seed = await loop.run_in_executor(
                worker, engine.synthesize, body.text, VOICES / f"{body.voice}.wav", output, voice["transcript"],
                body.engine, chatterbox_settings, body.xtts_license_accepted)
            result = {"audioUrl": f"/api/audio/{job_id}", "duration": duration}
            if body.engine == "chatterbox" and used_seed is not None:
                if chatterbox_settings is None:
                    chatterbox_settings = {}
                chatterbox_settings["seed"] = used_seed
                result["settings"] = chatterbox_settings
        jobs[job_id].update(state="done", seconds=round(time.monotonic() - started, 2), **result)
    except Exception as exc:
        logger.exception("Voice job failed")
        output.unlink(missing_ok=True)
        jobs[job_id].update(state="error", error=str(exc) if isinstance(exc, EngineError) else "Не удалось подготовить голос или создать аудио. "
                            "Проверьте интернет при первом скачивании модели и журнал сервера.")
    finally:
        generation_lock.release()


async def start_job(body=None):
    body = body or EngineRequest()
    if body.engine == "xtts" and XTTS_CONSENT.exists():
        body = body.model_copy(update={"xtts_license_accepted": True})
    if body.engine == "xtts" and not body.xtts_license_accepted:
        raise HTTPException(422, "Для XTTS ознакомьтесь с лицензией и подтвердите условия в интерфейсе.")
    if generation_lock.locked():
        raise HTTPException(409, "Уже выполняется запрос. Дождитесь завершения.")
    if body.engine == "xtts" and body.xtts_license_accepted and not XTTS_CONSENT.exists():
        XTTS_CONSENT.write_text(json.dumps({"model": ENGINES["xtts"]["model"],
            "license": "https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt", "accepted_at": time.time()}))
    await generation_lock.acquire()
    # Keep a bounded number of finished job records in memory.
    for old in list(jobs)[:-99]:
        if jobs[old]["state"] != "running":
            del jobs[old]
    job_id = uuid.uuid4().hex
    job_data = {"id": job_id, "state": "running", "engine": body.engine,
                "engineName": ENGINES[body.engine]["name"]}
    if body.engine == "chatterbox":
        job_data["settings"] = resolve_chatterbox_settings(body)
    jobs[job_id] = job_data
    task = asyncio.create_task(run_job(job_id, body))
    active_tasks.add(task)
    task.add_done_callback(active_tasks.discard)
    return jobs[job_id].copy()


active_tasks = set()


@app.post("/api/tts/prepare", status_code=202)
async def prepare(body: EngineRequest):
    return await start_job(body)


@app.post("/api/speech", status_code=202)
async def speech(body: SpeechRequest):
    get_voice(body.voice)
    return await start_job(body)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Задание не найдено. Возможно, сервер был перезапущен.")
    return jobs[job_id]


@app.get("/api/audio/{audio_id}")
def audio_file(audio_id: str):
    if not re.fullmatch(r"[a-f0-9]{32}", audio_id):
        raise HTTPException(404, "Аудио не найдено.")
    path = OUTPUTS / f"{audio_id}.wav"
    if not path.exists() or jobs.get(audio_id, {}).get("state") == "running":
        raise HTTPException(404, "Аудио ещё не готово или не найдено.")
    return FileResponse(path, media_type="audio/wav", filename=f"voice-lab-{audio_id[:8]}.wav")


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="ui")
