# Inspired from https://huggingface.co/spaces/vumichien/whisper-speaker-diarization/blob/main/app.py

import whisper
import datetime
import subprocess
import gradio as gr
from pathlib import Path
import pandas as pd
import re
import time
import os 
import numpy as np

from pytube import YouTube
import torch
# import pyannote.audio
# from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
# from pyannote.audio import Audio
# from pyannote.core import Segment
# from sklearn.cluster import AgglomerativeClustering

from gpuinfo import GPUInfo

import wave
import contextlib
from transformers import pipeline
import psutil

# Custom code
from bechdelaidemo.utils import download_youtube_video
from bechdelaidemo.utils import extract_audio_from_movie

# Constants
whisper_models = ["tiny.en","base.en","tiny","base", "small", "medium", "large"]
device = 0 if torch.cuda.is_available() else "cpu"
os.makedirs('output', exist_ok=True)

# Prepare embedding model
# embedding_model = PretrainedSpeakerEmbedding( 
#     "speechbrain/spkrec-ecapa-voxceleb",
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def get_youtube(video_url):
    yt = YouTube(video_url)
    abs_video_path = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    print("Success download video")
    print(abs_video_path)
    return abs_video_path

def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str


def speech_to_text(video_filepath, selected_source_lang = "en", whisper_model = "tiny.en"):
    """
    # Transcribe youtube link using OpenAI Whisper
    1. Using Open AI's Whisper model to seperate audio into segments and generate transcripts.
    2. Generating speaker embeddings for each segments.
    3. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
    
    Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper
    Speaker diarization model and pipeline from by https://github.com/pyannote/pyannote-audio
    """
    
    time_start = time.time()

    # Convert video to audio
    audio_filepath = extract_audio_from_movie(video_filepath,".wav")

    # Load whisper
    model = whisper.load_model(whisper_model)

    # Get duration
    with contextlib.closing(wave.open(audio_filepath,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    print(f"conversion to wav ready, duration of audio file: {duration}")

    # Transcribe audio
    options = dict(language=selected_source_lang, beam_size=5, best_of=5)
    transcribe_options = dict(task="transcribe", **options)
    result = model.transcribe(audio_filepath, **transcribe_options)
    segments = result["segments"]
    text = result["text"].strip()
    print("starting whisper done with whisper")

    return [text]

source_language_list = ["en","fr"]

# ---- Gradio Layout -----
# Inspiration from https://huggingface.co/spaces/RASMUS/Whisper-youtube-crosslingual-subtitles
video_in = gr.Video(label="Video file", mirror_webcam=False)
youtube_url_in = gr.Textbox(label="Youtube url", lines=1, interactive=True)
selected_source_lang = gr.Dropdown(choices=source_language_list, type="value", value="en", label="Spoken language in video", interactive=True)
selected_whisper_model = gr.Dropdown(choices=whisper_models, type="value", value="tiny.en", label="Selected Whisper model", interactive=True)
# transcription_df = gr.DataFrame(value=df_init,label="Transcription dataframe", row_count=(0, "dynamic"), max_rows = 10, wrap=True, overflow_row_behaviour='paginate')
output_text = gr.Textbox(label = "Transcribed text",lines = 10)

title = "BechdelAI - demo"
demo = gr.Blocks(title=title,live = True)
demo.encrypt = False


with demo:
    with gr.Tab("BechdelAI - dialogue demo"):
        gr.Markdown('''
            <div>
                <h1 style='text-align: center'>BechdelAI - Dialogue demo</h1>
            </div>
        ''')

        with gr.Row():
            gr.Markdown('''# 🎥 Download Youtube video''')
              

        with gr.Row():

            with gr.Column():
                # gr.Markdown('''### You can test by following examples:''')
                examples = gr.Examples(examples=
                        [
                        "https://www.youtube.com/watch?v=FDFdroN7d0w",
                        "https://www.youtube.com/watch?v=b2f2Kqt_KcE",
                        "https://www.youtube.com/watch?v=ba5F8G778C0",
                    ],
                    label="Examples", inputs=[youtube_url_in])
                youtube_url_in.render()
                download_youtube_btn = gr.Button("Download Youtube video")
                download_youtube_btn.click(get_youtube, [youtube_url_in], [
                    video_in])
                print(video_in)
                
            with gr.Column():
                video_in.render()

        with gr.Row():
            gr.Markdown('''# 🎙 Extract text from video''')

        with gr.Row():
            with gr.Column():
                selected_source_lang.render()
                selected_whisper_model.render()
                transcribe_btn = gr.Button("Transcribe audio and diarization")
                transcribe_btn.click(speech_to_text, [video_in, selected_source_lang, selected_whisper_model], [output_text])
            with gr.Column():
                output_text.render()

demo.launch(debug=True)