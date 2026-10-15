"""Clause-level text chunks for TTS: TTS can start on "Sure," while the LLM is still writing the rest."""

import re

from livekit.agents.tokenize import token_stream, tokenizer

_CLAUSE_END = re.compile(r"(?<=[,;:.!?])\s+")  # "1,000" has no space after the comma, so it stays whole
MIN_CHARS = 8  # shorter pieces ("Dr.", "Hi,") are merged into the next one


def split_clauses(text: str) -> list[tuple[str, int, int]]:
    spans, start = [], 0
    for m in _CLAUSE_END.finditer(text):
        spans.append((text[start : m.start()], start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((text[start:], start, len(text)))
    return spans


class ClauseTokenizer(tokenizer.SentenceTokenizer):
    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        return [span[0] for span in split_clauses(text)]

    def stream(self, *, language: str | None = None) -> tokenizer.SentenceStream:
        return token_stream.BufferedSentenceStream(tokenizer=split_clauses, min_token_len=MIN_CHARS, min_ctx_len=4)


assert [s for s, _, _ in split_clauses("Sure, I found 1,000 slots. Book one?")] == [
    "Sure,", "I found 1,000 slots.", "Book one?"
]
