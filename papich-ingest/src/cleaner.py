import re
from .models import validate_segments


def clean_segments(raw):
    """Conservative cleaning. Never suppress repeated speech at distinct timestamps."""
    cleaned = []
    for source in sorted(validate_segments(raw), key=lambda s: (s['start'], s['end'])):
        text = re.sub(r'\s+', ' ', source['text']).strip()
        if not text or source['end'] <= source['start']:
            continue
        item = {'start': source['start'], 'end': source['end'], 'text': text,
                'speaker': source.get('speaker')}
        if cleaned and all(item[k] == cleaned[-1][k] for k in ('start', 'end', 'text', 'speaker')):
            continue
        # Merge only a tiny preceding segment of the same speaker, across <= 0.5s.
        if (cleaned and len(cleaned[-1]['text'].split()) < 4
                and 0 <= item['start'] - cleaned[-1]['end'] <= 0.5
                and item['speaker'] == cleaned[-1]['speaker']):
            cleaned[-1]['text'] += ' ' + text
            cleaned[-1]['end'] = item['end']
        else:
            cleaned.append(item)
    return cleaned
