import os
import tempfile
import whisper
# from pytube import YouTube
from pytubefix import YouTube
from pathlib import Path

YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=SOvZ-SxftV4" # Lore Recap Before DLC
# YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=Nlth97VZP70" # Rellana, Twin Moon Knight

DIR = Path(__file__).parent / "transcription/transcription.txt"

# Let's do this only if we haven't created the transcription file yet.
if not os.path.exists(DIR):
    youtube = YouTube(YOUTUBE_VIDEO)
    audio = youtube.streams.filter(only_audio=True).first()
    
    whisper_model = whisper.load_model("base.en")

    with tempfile.TemporaryDirectory() as tmpdir:
        file = audio.download(output_path=tmpdir)
        transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

        with open(DIR, "w") as file:
            file.write(transcription)