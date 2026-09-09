import whisper
import os


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None


def load_model():
    global _model

    if _model is None:
        print("Loading Whisper model...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")

    return _model


def transcribe(
    chunk_path: str,
    chunk_id: int,
    source: str = "audio",
    translate: bool = False
) -> list:

    model = load_model()

    task = "translate" if translate else "transcribe"

    result = model.transcribe(
        chunk_path,
        task=task
    )

    # Each chunk is 10 minutes = 600 seconds
    chunk_duration = 10 * 60

    # Calculate the starting time of this chunk
    chunk_offset = chunk_id * chunk_duration

    segments = []

    for segment in result["segments"]:

        # Convert chunk-relative time
        # into full-audio time
        start = segment["start"] + chunk_offset
        end = segment["end"] + chunk_offset

        segments.append(
            {
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
                "text": segment["text"].strip(),
                "source": source
            }
        )

    return segments


def transcribe_all(
    chunks: list,
    source: str = "audio",
    translate: bool = False
) -> list:

    all_segments = []

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk "
            f"{i + 1}/{len(chunks)}..."
        )

        segments = transcribe(
            chunk_path=chunk,
            chunk_id=i,
            source=source,
            translate=translate
        )

        all_segments.extend(segments)

    print("Transcription completed.")

    return all_segments