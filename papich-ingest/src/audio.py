from pathlib import Path
import shutil
import subprocess
import wave


def check_ffmpeg():
    if not shutil.which('ffmpeg'):
        raise RuntimeError('ffmpeg not found. Install: brew install ffmpeg')


def duration(path):
    with wave.open(str(path), 'rb') as src:
        if src.getnchannels() != 1 or src.getframerate() != 16000 or src.getsampwidth() != 2:
            raise ValueError('Expected mono 16 kHz PCM16 WAV')
        frames = src.getnframes()
        if frames <= 0 or Path(path).stat().st_size < frames * 2 + 44:
            raise ValueError('Empty or truncated WAV')
        return frames / 16000


def normalize(source, output):
    check_ffmpeg()
    output = Path(output)
    temp = output.with_suffix('.partial.wav')
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
               '-i', str(source), '-vn', '-ac', '1', '-ar', '16000',
               '-c:a', 'pcm_s16le', str(temp)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('ffmpeg normalization failed: ' + result.stderr[-3000:])
    duration(temp)
    temp.replace(output)


def extract_block(source, output, start, seconds):
    with wave.open(str(source), 'rb') as src:
        src.setpos(round(start * 16000))
        with wave.open(str(output), 'wb') as dst:
            dst.setparams(src.getparams())
            dst.writeframes(src.readframes(round(seconds * 16000)))
