from pathlib import Path
import logging
from .models import Video
from .urls import canonical_url, video_id, playlist_id

log = logging.getLogger(__name__)


def options(config):
    opts = {'quiet': True, 'no_warnings': False, 'logger': log,
            'noplaylist': True, 'retries': 3, 'fragment_retries': 3,
            'socket_timeout': 30, 'cachedir': False}
    if config.cookies_file:
        opts['cookiefile'] = config.cookies_file
    return opts


def expand_url(url, config, playlists=False):
    if video_id(url) and not (playlists and playlist_id(url)):
        return [url]
    if not playlist_id(url):
        raise ValueError(f'Expected YouTube video or playlist URL: {url}')
    if not playlists:
        raise ValueError('Playlist URL requires --playlist')
    import yt_dlp
    opts = options(config) | {'extract_flat': 'in_playlist', 'noplaylist': False,
                              'ignoreerrors': True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise RuntimeError('Could not read playlist')
    urls = []
    for entry in info.get('entries', []):
        if entry and video_id(entry.get('id', '')):
            urls.append(canonical_url(entry['id']))
        else:
            # yt-dlp can omit entries whose identifiers YouTube does not expose.
            log.warning('Playlist contains an inaccessible entry without a usable video id')
    return urls


def metadata(url, config):
    import yt_dlp
    with yt_dlp.YoutubeDL(options(config)) as ydl:
        info = ydl.extract_info(canonical_url(url), download=False)
    if not info:
        raise RuntimeError('yt-dlp returned no video metadata')
    if info.get('is_live') or info.get('live_status') == 'is_upcoming':
        raise ValueError('Live/upcoming broadcasts are unsupported; retry after publication')
    return Video(id=info['id'], title=info.get('title') or info['id'],
                 channel=info.get('channel') or info.get('uploader') or '',
                 url=canonical_url(info['id']), original_url=url,
                 upload_date=info.get('upload_date'), duration=info.get('duration'))


def download(video, directory, config):
    import yt_dlp
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    opts = options(config) | {'format': 'bestaudio/best[vcodec=none]',
                              'outtmpl': str(directory / 'source.%(ext)s')}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video.url, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError('Audio download produced no file')
    return path
