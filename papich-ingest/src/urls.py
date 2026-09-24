import re
from urllib.parse import parse_qs, urlparse

VIDEO_ID = re.compile(r'^[A-Za-z0-9_-]{11}$')
HOSTS = {'youtube.com', 'www.youtube.com', 'm.youtube.com', 'music.youtube.com',
         'youtube-nocookie.com', 'www.youtube-nocookie.com'}


def video_id(url):
    if VIDEO_ID.fullmatch(url):
        return url
    parsed = urlparse(url if '://' in url else 'https://' + url)
    host = (parsed.hostname or '').lower()
    parts = parsed.path.strip('/').split('/')
    candidate = None
    if host in {'youtu.be', 'www.youtu.be'}:
        candidate = parts[0]
    elif host in HOSTS:
        if parsed.path == '/watch':
            candidate = parse_qs(parsed.query).get('v', [None])[0]
        elif len(parts) == 2 and parts[0] in {'shorts', 'embed', 'live', 'v'}:
            candidate = parts[1]
    return candidate if candidate and VIDEO_ID.fullmatch(candidate) else None


def canonical_url(value):
    key = video_id(value)
    if not key:
        raise ValueError(f'Not a YouTube video URL/id: {value}')
    return f'https://www.youtube.com/watch?v={key}'


def timestamp_url(value, seconds):
    return canonical_url(value) + f'&t={max(0, int(seconds))}s'


def playlist_id(url):
    p = urlparse(url)
    return parse_qs(p.query).get('list', [None])[0] if p.hostname in HOSTS else None
