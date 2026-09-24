from dataclasses import dataclass, asdict
import math


@dataclass
class Video:
    id: str
    title: str
    channel: str
    url: str
    original_url: str
    upload_date: str | None = None
    duration: float | None = None

    def to_dict(self):
        return asdict(self)


def validate_segments(segments):
    if not isinstance(segments, list):
        raise ValueError('segments must be a list')
    for s in segments:
        if (not isinstance(s.get('text'), str) or
                not all(isinstance(s.get(k), (int, float)) and math.isfinite(s[k])
                        for k in ('start', 'end')) or
                not 0 <= s['start'] <= s['end']):
            raise ValueError(f'Invalid speech segment: {s!r}')
    return segments
