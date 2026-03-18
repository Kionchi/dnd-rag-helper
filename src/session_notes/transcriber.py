from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class TranscriptSegment:
    text: str
    start_sec: float
    end_sec: float


class Transcriber(ABC):
    """Abstract interface for speech-to-text backends."""

    @abstractmethod
    def transcribe_stream(self) -> Iterable[TranscriptSegment]:
        """Yield transcript segments from a live or simulated audio stream."""
        raise NotImplementedError


class DummyTranscriber(Transcriber):
    """Placeholder implementation for development without real audio."""

    def __init__(self, scripted_segments: Optional[List[TranscriptSegment]] = None) -> None:
        self._segments = scripted_segments or []

    def transcribe_stream(self) -> Iterable[TranscriptSegment]:
        for seg in self._segments:
            yield seg