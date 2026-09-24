from .urls import timestamp_url


def split_segment(segment, fits):
    """Split only oversized speech segments, retaining their original time bounds."""
    words = segment['text'].split()
    pieces = []
    while words:
        low, high = 1, len(words)
        best = 0
        while low <= high:
            middle = (low + high) // 2
            if fits(' '.join(words[:middle])):
                best, low = middle, middle + 1
            else:
                high = middle - 1
        if best == 0:
            # A single pathological ASR token can itself exceed the model context.
            word = words.pop(0)
            cut = len(word) // 2
            if cut == 0:
                raise ValueError('Tokenizer budget too small for a single character')
            words[:0] = [word[:cut], word[cut:]]
            continue
        pieces.append(dict(segment, text=' '.join(words[:best])))
        words = words[best:]
    if len(pieces) > 1:
        for piece in pieces:
            piece['timing_scope'] = 'original_segment'
    return pieces


def chunk_segments(segments, video, config, count_tokens):
    """count_tokens must include passage prefix and tokenizer special tokens."""
    fits = lambda text: count_tokens(text) <= config.chunk_max_tokens
    parts = [piece for segment in segments for piece in split_segment(segment, fits)]
    chunks, start = [], 0
    while start < len(parts):
        end, text = start, ''
        while end < len(parts):
            proposed = (text + ' ' + parts[end]['text']).strip()
            if not fits(proposed):
                break
            text = proposed
            end += 1
            if len(text.split()) >= config.chunk_target_words:
                break
        if end == start:
            raise ValueError('Oversized chunk after segment splitting')
        selected = parts[start:end]
        speakers = sorted({s['speaker'] for s in selected if s.get('speaker')})
        index = len(chunks)
        begin, finish = min(s['start'] for s in selected), max(s['end'] for s in selected)
        chunks.append({
            'id': f"{video['id']}_{index:06d}", 'video_id': video['id'],
            'title': video['title'], 'channel': video['channel'],
            'upload_date': video.get('upload_date'), 'source_url': video['url'],
            'original_url': video.get('original_url', video['url']),
            'timestamp_url': timestamp_url(video['id'], begin),
            'start': begin, 'end': finish,
            'speaker': speakers[0] if len(speakers) == 1 and all(s.get('speaker') for s in selected) else None,
            'speakers': speakers, 'text': text, 'chunk_index': index,
            'token_count': count_tokens(text), 'duplicate_of': None, 'duplicate_score': None,
            'timing_scope': 'speech_segments',
        })
        if end == len(parts):
            break
        # Whole-segment overlap, capped by token budget; always consume new segments.
        next_start = end
        while next_start > start + 1 and config.chunk_overlap_tokens:
            tail = ' '.join(p['text'] for p in parts[next_start - 1:end])
            if count_tokens(tail) > config.chunk_overlap_tokens:
                break
            next_start -= 1
        # Shrink overlap when the next segment is large: never emit an overlap-only chunk.
        while next_start < end and not fits(' '.join(p['text'] for p in parts[next_start:end + 1])):
            next_start += 1
        start = next_start
    return chunks
