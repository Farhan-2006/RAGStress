import re
import unicodedata

# Preserve negation and biomedical words: no stemming or stopword removal.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")

def tokenize(text):
    return TOKEN_PATTERN.findall(unicodedata.normalize('NFKC', text).casefold())

def sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text) if s.strip()]

def chunks(document, max_words=150):
    """Bounded sentence windows with one-sentence overlap, same source ID.

    Dense encoding also applies its tokenizer length limit (model default 256).
    Extremely long sentences are split into non-overlapping word windows.
    """
    pieces = []
    for sentence in sentences(document['text']):
        words = sentence.split()
        pieces.extend(' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words))
    windows, current, count = [], [], 0
    for piece in pieces:
        size = len(piece.split())
        if current and count + size > max_words:
            windows.append(' '.join(current))
            current = [current[-1]] if len(current[-1].split()) + size <= max_words else []
            count = sum(len(s.split()) for s in current)
        current.append(piece)
        count += size
    if current:
        windows.append(' '.join(current))
    return [{'id': f"{document['id']}:{i}", 'source_id': document['id'], 'text': text,
             'title': document['title']} for i, text in enumerate(windows or [document['text']])]
