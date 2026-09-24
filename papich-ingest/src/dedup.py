import re
import unicodedata


def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def shingles(text):
    words = normalize(text).split()
    return {tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))}


def similarity(left, right):
    a, b = normalize(left), normalize(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    x, y = shingles(a), shingles(b)
    return len(x & y) / len(x | y) if x and y else 0.0


def annotate(chunks, candidates, threshold=0.92):
    """Advisory lexical cross-video matching. No match is deleted or hidden."""
    for chunk in chunks:
        chunk['duplicate_of'], chunk['duplicate_score'] = None, None
    # Streaming DB scan bounds memory by current video's chunks.
    for candidate in candidates:
        for chunk in chunks:
            score = similarity(chunk['text'], candidate['text'])
            if score >= threshold and score > (chunk['duplicate_score'] or 0):
                chunk['duplicate_of'], chunk['duplicate_score'] = candidate['id'], score
    for index, chunk in enumerate(chunks):
        for previous in chunks[:index]:
            score = similarity(chunk['text'], previous['text'])
            if score >= threshold and score > (chunk['duplicate_score'] or 0):
                chunk['duplicate_of'], chunk['duplicate_score'] = previous['id'], score
    return chunks
