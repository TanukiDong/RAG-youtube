import os
import tempfile
import whisper
# from pytube import YouTube
from pytubefix import YouTube

YOUTUBE_VIDEO = """https://www.youtube.com/watch?v=SOvZ-SxftV4""" # Lore Recap Before DLC
# YOUTUBE_VIDEO = "https://www.youtube.com/watch?v=Nlth97VZP70" # Rellana, Twin Moon Knight
# Define the path to the transcription file in the "src" folder
transcription_file_path = os.path.join("src/youtube-rag", "transcription.txt")

# Let's do this only if we haven't created the transcription file yet.
if not os.path.exists(transcription_file_path):
    youtube = YouTube(YOUTUBE_VIDEO)
    audio = youtube.streams.filter(only_audio=True).first()
    
    whisper_model = whisper.load_model("medium.en")

    with tempfile.TemporaryDirectory() as tmpdir:
        file = audio.download(output_path=tmpdir)
        transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

        with open(transcription_file_path, "w") as file:
            file.write(transcription)