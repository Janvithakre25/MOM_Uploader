"""
utils/audio_processor.py

Handles:
  1. Downloading an audio file from Supabase Storage to a temp path.
  2. Splitting it into safe-sized chunks using pydub.
  3. Sending each chunk to the Groq Whisper API sequentially.
  4. Concatenating all partial transcripts into one master transcript.
"""

import os
import uuid
import logging
import tempfile
from pathlib import Path

from pydub import AudioSegment
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
# 10 minutes per chunk by default (600 000 ms).
# Groq's free tier allows up to ~25 MB per request; 10-min MP3 ≈ 10 MB @ 128 kbps.
MAX_CHUNK_DURATION_MS: int = int(os.getenv("MAX_CHUNK_DURATION_MS", 600_000))
TEMP_AUDIO_DIR: str = os.getenv("TEMP_AUDIO_DIR", tempfile.gettempdir())

# Supported formats pydub can decode (requires ffmpeg for mp3/m4a/ogg)
SUPPORTED_FORMATS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_temp_dir() -> Path:
    """Create the temporary working directory if it doesn't exist."""
    path = Path(TEMP_AUDIO_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_audio(file_path: str) -> tuple[AudioSegment, str]:
    """
    Load an audio file into a pydub AudioSegment.
    Returns the segment and the detected file extension.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    # pydub uses ffmpeg under the hood for mp3/m4a/ogg/flac/webm
    fmt = ext.lstrip(".")
    if fmt == "mp3":
        audio = AudioSegment.from_mp3(file_path)
    elif fmt == "wav":
        audio = AudioSegment.from_wav(file_path)
    elif fmt == "ogg":
        audio = AudioSegment.from_ogg(file_path)
    else:
        audio = AudioSegment.from_file(file_path, format=fmt)

    logger.info(
        "Loaded audio: %.1f minutes (%d ms), %d channels, %d Hz",
        len(audio) / 60_000,
        len(audio),
        audio.channels,
        audio.frame_rate,
    )
    return audio, fmt


def split_audio_into_chunks(
    file_path: str,
    chunk_duration_ms: int = MAX_CHUNK_DURATION_MS,
) -> list[Path]:
    """
    Split an audio file into chunks of at most `chunk_duration_ms` milliseconds.

    Args:
        file_path: Absolute path to the source audio file.
        chunk_duration_ms: Maximum duration of each chunk in milliseconds.

    Returns:
        Ordered list of Path objects pointing to the temporary chunk files.
        The caller is responsible for deleting these files after use.
    """
    temp_dir = _ensure_temp_dir()
    audio, fmt = _load_audio(file_path)

    total_ms = len(audio)
    chunk_paths: list[Path] = []

    if total_ms <= chunk_duration_ms:
        # Single chunk — no splitting needed; copy as-is to avoid re-encoding.
        chunk_path = temp_dir / f"chunk_0_{uuid.uuid4().hex}.{fmt}"
        audio.export(str(chunk_path), format=fmt)
        chunk_paths.append(chunk_path)
        logger.info("Audio fits in a single chunk: %s", chunk_path.name)
        return chunk_paths

    # Split into multiple overlapping-free slices
    num_chunks = -(-total_ms // chunk_duration_ms)  # ceiling division
    logger.info(
        "Splitting %d ms audio into %d chunks of ≤%d ms",
        total_ms,
        num_chunks,
        chunk_duration_ms,
    )

    for i in range(num_chunks):
        start_ms = i * chunk_duration_ms
        end_ms = min(start_ms + chunk_duration_ms, total_ms)
        chunk = audio[start_ms:end_ms]

        chunk_path = temp_dir / f"chunk_{i}_{uuid.uuid4().hex}.mp3"
        # Export as low-bitrate MP3 to guarantee we stay well under Groq's 25MB limit
        chunk.export(str(chunk_path), format="mp3", parameters=["-b:a", "64k"])
        chunk_paths.append(chunk_path)
        logger.info(
            "  Chunk %d/%d: %s  (%.1f min)",
            i + 1,
            num_chunks,
            chunk_path.name,
            (end_ms - start_ms) / 60_000,
        )

    return chunk_paths


def _transcribe_chunk(groq_client: Groq, chunk_path: Path) -> str:
    """
    Send a single audio chunk to the Groq Whisper API and return the transcript.
    """
    with open(chunk_path, "rb") as audio_file:
        response = groq_client.audio.transcriptions.create(
            file=(chunk_path.name, audio_file),
            model="whisper-large-v3",
            response_format="text",
            temperature=0.0,        # Deterministic output
        )
    # When response_format="text", Groq returns a plain string
    if isinstance(response, str):
        return response.strip()
    # Fallback for object response
    return str(response).strip()


def transcribe_audio_file(file_path: str) -> str:
    """
    Full pipeline: split audio → transcribe each chunk → join into master transcript.

    Args:
        file_path: Absolute path to the uploaded audio file.

    Returns:
        Complete concatenated transcript as a single string.

    Raises:
        ValueError: If the file format is unsupported.
        groq.APIError: If the Groq API call fails.
    """
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    chunk_paths: list[Path] = []

    try:
        chunk_paths = split_audio_into_chunks(file_path)
        transcript_parts: list[str] = []

        for idx, chunk_path in enumerate(chunk_paths, start=1):
            logger.info(
                "Transcribing chunk %d/%d: %s", idx, len(chunk_paths), chunk_path.name
            )
            part = _transcribe_chunk(groq_client, chunk_path)
            transcript_parts.append(part)
            logger.info("  → %d characters transcribed", len(part))

        master_transcript = "\n\n".join(transcript_parts)
        logger.info(
            "Transcription complete: %d characters total", len(master_transcript)
        )
        return master_transcript

    finally:
        # Always clean up temporary chunk files
        for p in chunk_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("Could not delete temp chunk %s: %s", p, e)