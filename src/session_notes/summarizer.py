from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class SessionNote:
    text: str
    start_sec: float
    end_sec: float


class SessionSummarizer(ABC):
    """Abstract interface for summarizing raw transcript into concise notes."""

    @abstractmethod
    def summarize(self, raw_segments: List[str], start_sec: float, end_sec: float) -> SessionNote:
        """Summarize a window of transcript into a single note."""
        raise NotImplementedError


class HeuristicSummarizer(SessionSummarizer):
    """Very simple baseline: join lines and truncate. Replace later with LLM-based one."""

    def __init__(self, max_chars: int = 300) -> None:
        self.max_chars = max_chars

    def summarize(self, raw_segments: List[str], start_sec: float, end_sec: float) -> SessionNote:
        combined = " ".join(s.strip() for s in raw_segments if s.strip())
        if len(combined) > self.max_chars:
            combined = combined[: self.max_chars].rstrip() + "..."
        return SessionNote(text=combined, start_sec=start_sec, end_sec=end_sec)