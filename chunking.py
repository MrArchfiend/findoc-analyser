from __future__ import annotations

import re
import nltk
from dataclasses import dataclass, field
from typing import Optional


# Registry of (regex, section_name) pairs for 10-K section detection.
# Add new tuples here to support additional section headers.
_SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\brisk factors?\b',                 re.I), "Risk Factors"),
    (re.compile(r'\bmanagement.{0,10}discussion\b',   re.I), "MD&A"),
    (re.compile(r'\bfinancial statements?\b',          re.I), "Financial Statements"),
    (re.compile(r'\bproperties\b',                    re.I), "Properties"),
    (re.compile(r'\blegal proceedings?\b',             re.I), "Legal Proceedings"),
    (re.compile(r'\bquantitative.{0,20}qualitative\b', re.I), "Market Risk"),
    (re.compile(r'\bcontrols? and procedures?\b',     re.I), "Controls & Procedures"),
    (re.compile(r'\bexecutive compensation\b',         re.I), "Executive Compensation"),
    (re.compile(r'\bsecurity ownership\b',             re.I), "Security Ownership"),
]


# Dataclass holding a text segment and its retrieval metadata.
# Add caller-specific fields (e.g. source file, page number) to `extra`.
@dataclass
class Chunk:
    text: str
    chunk_index: int
    method: str
    char_start: int
    char_end: int
    section_hint: Optional[str] = None
    extra: dict = field(default_factory=dict)

    # Return a one-line debug summary showing index, char range, section, and text preview.
    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(index={self.chunk_index}, "
            f"chars={self.char_start}-{self.char_end}, "
            f"section={self.section_hint!r}, "
            f"preview={preview!r})"
        )


class ParagraphAwareChunker:

    # Store config and download the NLTK punkt tokeniser once at construction time.
    def __init__(self, max_chars: int = 1500, overlap_sentences: int = 2) -> None:
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        self._sent_tokenize = sent_tokenize

    # Public entry point — route each paragraph to PATH A/B (merge) or PATH C (sentence-split),
    # then wrap resulting segments into Chunk objects.
    def chunk(self, text: str) -> list[Chunk]:
        paragraphs = self._parse_paragraphs(text)

        merge_buffer: list[str] = []
        merge_len: int = 0
        segments: list[str] = []

        for para in paragraphs:
            para_len = len(para)

            if para_len > self.max_chars:
                # PATH C — oversized paragraph: flush buffer then sentence-split.
                merge_len = self._flush_merge_buffer(merge_buffer, segments)
                segments.extend(self._split_large_paragraph(para))

            else:
                # PATH A/B — fits within limit: merge with buffered neighbours.
                if merge_buffer and merge_len + para_len + 2 > self.max_chars:
                    last_para = merge_buffer[-1]
                    merge_len = self._flush_merge_buffer(merge_buffer, segments)
                    merge_buffer.append(last_para)   # carry last para as overlap
                    merge_len = len(last_para)

                merge_buffer.append(para)
                merge_len += para_len + 2

        self._flush_merge_buffer(merge_buffer, segments)
        return self._build_chunks(segments)


  # TASK_1: define a function _parse_paragraphs
    # Split document text on \n\n boundaries into a list of clean paragraph strings.


  # TASK_2: define a function _flush_merge_buffer
    # Join buffered paragraphs into one segment, append to segments, and clear the buffer.
    # Returns 0 so the caller can reset merge_len in a single assignment


  # TASK_3: define a function _split_large_paragraph
    # Split an oversized paragraph into sentence-grouped sub-chunks with sentence overlap.
    # If a single sentence exceeds max_chars, keep it intact rather than splitting mid-sentence.


  # TASK_4: define a function _build_chunks
    # Wrap raw text segments into Chunk objects with char offsets and section labels.


  # TASK_5: define a function _detect_section
    # Scan the first 300 chars of text against _SECTION_PATTERNS and return the first match.
  
