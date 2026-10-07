import re
from .preprocessing import sentences

def extract_claims(answer, limit=3):
    """Conservative sentence/semicolon candidates; not a perfect atomic parser.

    Do not split on 'and': biomedical comparisons often require both clauses.
    A complex sentence is retained whole and visibly identified as a candidate.
    """
    text = re.sub(r'\[[^\]]+\]', '', answer)
    candidates = [piece.strip(' \n-*') for sentence in sentences(text)
                  for piece in sentence.split(';') if piece.strip()]
    return candidates[:limit], len(candidates) > limit
