import yt_dlp
from pydub import AudioSegment
import os

DONWLOAD_DIR= 'donwloades'
os.makedirs(DONWLOAD_DIR,exist_ok=True)

def donwload_youtube_aduio (url :str) -> str:
    output_path=os.path.join(DONWLOAD_DIR,"%(title)s.%(ext)s")
    ydl_opts ={
        "format":"bestaudio/best",
        "outtmpl":output_path,
        "postprocessors":[
            {
                "key":"FFmpegExtractAudio",
                "preferredcodec":"wav",
                "preferredquality":"192",
            }
        ],
        "quiet":True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info =ydl.extract_info(url,download=True)
        filename=ydl.prepare_filename(info).replace(".webm",".wav").replace(".m4a",".wav")
        audio = AudioSegment.from_file(filename)
        audio =audio.set_channels(1).set_frame_rate(16000)
        audio.export(filename, format="wav")
    return filename


def convert_to_wav(input_path :str) ->str:
    output_path =os.path.splitext(input_path)[0] + "converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio =audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path,format="wav")
    return output_path



def chunk_audio(wav_path: str, chunk_min: int = 10) -> list:
    audio = AudioSegment.from_file(wav_path)

    if len(audio) == 0:
        raise ValueError("Audio file is empty. Whisper cannot transcribe it.")

    chunk_ms = chunk_min * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):

        chunk = audio[start:start + chunk_ms]

        # Skip empty chunks
        if len(chunk) == 0:
            continue

        chunk_path = f"{wav_path}_chunk{i}.wav"

        # Make sure audio is Whisper-compatible
        chunk = (
            chunk
            .set_channels(1)
            .set_frame_rate(16000)
            .set_sample_width(2)
        )

        chunk.export(
            chunk_path,
            format="wav"
        )

        chunks.append(chunk_path)

    if not chunks:
        raise ValueError("No valid audio chunks were created.")

    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = donwload_youtube_aduio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)
 
    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks