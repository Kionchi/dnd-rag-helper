"""
CLI runner for live (or simulated) session notes capture.

MVP: uses DummyTranscriber + HeuristicSummarizer and writes notes to:
- logs/session_current.md
- Chroma collection: session_notes
"""

from pathlib import Path
from typing import List

from .transcriber import DummyTranscriber, TranscriptSegment
from .summarizer import HeuristicSummarizer
from .storage import append_note_to_markdown, upsert_note_into_chroma


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "logs" / "session_current.md"
DB_PATH = PROJECT_ROOT / "db"
SYSTEM_NAME = "D&D 5e"
SESSION_ID = "default_session"


def _simulate_segments() -> List[TranscriptSegment]:
    return [
        TranscriptSegment("The party enters the old keep and meets a nervous guard.", 0.0, 30.0),
        TranscriptSegment("They negotiate safe passage in exchange for clearing the basement.", 30.0, 60.0),
        TranscriptSegment("The wizard casts Fireball on the swarm of rats.", 60.0, 90.0),
    ]


def main() -> None:
    print("Starting session notes runner (MVP, simulated transcript)...")

    transcriber = DummyTranscriber(scripted_segments=_simulate_segments())
    summarizer = HeuristicSummarizer(max_chars=280)

    current_window: List[str] = []
    window_start: float | None = None
    window_end: float | None = None
    window_max_duration = 60.0  # seconds per note window

    for seg in transcriber.transcribe_stream():
        if window_start is None:
            window_start = seg.start_sec
        window_end = seg.end_sec
        current_window.append(seg.text)

        # flush window when enough time has passed
        if window_end - window_start >= window_max_duration:
            note = summarizer.summarize(current_window, start_sec=window_start, end_sec=window_end)
            append_note_to_markdown(note, LOG_PATH)
            upsert_note_into_chroma(
                note=note,
                system_name=SYSTEM_NAME,
                session_id=SESSION_ID,
                db_path=DB_PATH,
            )
            print(f"Wrote note [{note.start_sec:.0f}s–{note.end_sec:.0f}s]: {note.text}")
            current_window = []
            window_start = None
            window_end = None

    # flush remainder
    if current_window and window_start is not None:
        note = summarizer.summarize(current_window, start_sec=window_start, end_sec=window_end or window_start)
        append_note_to_markdown(note, LOG_PATH)
        upsert_note_into_chroma(
            note=note,
            system_name=SYSTEM_NAME,
            session_id=SESSION_ID,
            db_path=DB_PATH,
        )
        print(f"Wrote final note [{note.start_sec:.0f}s–{note.end_sec:.0f}s]: {note.text}")


if __name__ == "__main__":
    main()